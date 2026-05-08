import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
from components.episode_buffer import EpisodeBatch
from modules.mixers.abim_mixer import ABIMMixer


class IntrinsicMotivationModule(nn.Module):
    def __init__(self, repr_dim, action_dim, hidden_dim=128):
        super(IntrinsicMotivationModule, self).__init__()
        self.forward_model = nn.Sequential(
            nn.Linear(repr_dim + action_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, repr_dim)
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(repr_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, phi_t, phi_next, action_t):
        phi_next_pred = self.forward_model(torch.cat([phi_t, action_t], dim=-1))
        action_pred = self.inverse_model(torch.cat([phi_t, phi_next], dim=-1))
        with torch.no_grad():
            intrinsic_reward = F.mse_loss(phi_next_pred, phi_next, reduction='none').mean(dim=-1, keepdim=True)
        return phi_next_pred, action_pred, intrinsic_reward


class ABIMLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        act_total_dim = sum(args.action_dims_list)
        self.intrinsic_mods = nn.ModuleList([
            IntrinsicMotivationModule(args.embed_dim, act_total_dim) for _ in range(args.n_agents)
        ]).to(args.device)
        self.params += list(self.intrinsic_mods.parameters())

        self.mixer = ABIMMixer(args).to(args.device)
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.target_mac = copy.deepcopy(mac)
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        mac_out, phi_list, micro_qs_list = [], [], []
        self.mac.init_hidden(batch.batch_size)

        for t in range(batch.max_seq_length):
            tf_actions = batch["actions"][:, t] if t < batch.max_seq_length - 1 else None
            outs = self.mac.forward(batch, t=t, teacher_forcing_actions=tf_actions)
            mac_out.append(outs["top_q"])
            phi_list.append(outs["phi"])
            if t < batch.max_seq_length - 1:
                micro_qs_list.append(outs["micro_qs"])

        mac_out = torch.stack(mac_out, dim=1)
        phis = torch.stack(phi_list, dim=1)

        loss_seq_align, loss_representation = 0.0, 0.0

        # 🌟 优化 1：单独收集每个智能体的内在奖励，杜绝全局平均
        ind_int_rewards_list = []

        for i in range(self.args.n_agents):
            phi_t = phis[:, :-1, i, :]
            phi_next = phis[:, 1:, i, :].detach()

            act_i = actions[:, :, i, :]
            act_i_onehot = []
            for step, dim in enumerate(self.args.action_dims_list):
                act_i_onehot.append(F.one_hot(act_i[:, :, step].long(), num_classes=dim).float())
            act_i_flat = torch.cat(act_i_onehot, dim=-1)

            phi_next_pred, action_pred, int_reward = self.intrinsic_mods[i](phi_t, phi_next, act_i_flat)

            # 记录独立的好奇心数值 (B, T-1, 1)
            ind_int_rewards_list.append(int_reward)

            loss_representation += F.mse_loss(phi_next_pred, phi_next) + \
                                   self.args.inverse_lambda * F.mse_loss(action_pred, act_i_flat)

            top_q_val = mac_out[:, :-1, i].detach()
            for t in range(batch.max_seq_length - 1):
                mq_t = micro_qs_list[t]
                for step_idx in range(len(self.args.action_dims_list)):
                    if isinstance(mq_t, list) and len(mq_t) > 0 and isinstance(mq_t[0], list):
                        agent_micro_q = mq_t[i][step_idx]
                    elif isinstance(mq_t, list) and len(mq_t) > 0 and isinstance(mq_t[0], torch.Tensor):
                        tensor_q = mq_t[step_idx]
                        if tensor_q.dim() == 2:
                            agent_micro_q = tensor_q.view(batch.batch_size, self.args.n_agents, -1)[:, i, :]
                        else:
                            agent_micro_q = tensor_q[:, i, :]
                    else:
                        raise ValueError("Unknown micro_qs format")

                    act_target_idx = actions[:, t, i, step_idx].long().unsqueeze(-1)
                    micro_q_chosen = agent_micro_q.gather(1, act_target_idx).squeeze(-1)
                    loss_seq_align += F.mse_loss(micro_q_chosen, top_q_val[:, t])

        # 拼接成张量 (B, T-1, n_agents)
        ind_int_rewards = torch.cat(ind_int_rewards_list, dim=-1)

        with torch.no_grad():
            self.target_mac.init_hidden(batch.batch_size)
            target_mac_out = []
            for t in range(batch.max_seq_length):
                t_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(t_outs["top_q"])
            target_mac_out = torch.stack(target_mac_out[1:], dim=1)

        q_tot_current = self.mixer(mac_out[:, :-1], batch["state"][:, :-1])
        with torch.no_grad():
            q_tot_target = self.target_mixer(target_mac_out, batch["state"][:, 1:])

        # 🌟 优化 2：纯净的外部 TD Target，Mixer 仅拟合团队真实得分
        td_target = rewards + self.args.gamma * (1 - terminated) * q_tot_target
        td_error = (q_tot_current - td_target.detach())
        masked_td_error = td_error * mask
        loss_critic = (masked_td_error ** 2).sum() / mask.sum()

        # 🌟 优化 3：个体 Q 值整形 (Individual Q-Shaping)
        # 用各自产生的内在奖励直接乘以自己的 Q 值并取负。
        # 优化器在最小化该 Loss 时，会自动推高那些好奇心强的智能体的动作 Q 值，实现精准个体激励。
        masked_mac_out = mac_out[:, :-1] * mask
        loss_intrinsic_drive = - (masked_mac_out * ind_int_rewards.detach()).sum() / mask.sum()

        # 整体损失合成
        total_loss = loss_critic + self.args.intrinsic_beta * loss_intrinsic_drive + \
                     loss_seq_align / (batch.max_seq_length * self.args.n_agents) + loss_representation

        self.optimiser.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", total_loss.item(), t_env)
            self.logger.log_stat("critic_loss", loss_critic.item(), t_env)
            self.logger.log_stat("seq_loss", loss_seq_align.item(), t_env)
            self.logger.log_stat("repr_loss", loss_representation.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)

            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (q_tot_current * mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", (td_target * mask).sum().item() / mask_elems, t_env)

            # 记录独立好奇心的平均水平
            self.logger.log_stat("intrinsic_rewards_mean",
                                 (ind_int_rewards * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()
        self.intrinsic_mods.cuda()