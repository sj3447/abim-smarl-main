import torch
from .basic_controller import BasicMAC


class ABIMMAC(BasicMAC):
    def select_actions(self, ep_batch, t_ep, t_env, bs=None, test_mode=False):
        if hasattr(self.action_selector, "schedule"):
            self.action_selector.epsilon = self.action_selector.schedule.eval(t_env)

        epsilon = 0.0 if test_mode else self.action_selector.epsilon

        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, epsilon=epsilon)

        return agent_outputs["macro_idx"].squeeze(-1)

    def forward(self, ep_batch, t, test_mode=False, epsilon=0.0, teacher_forcing_actions=None):
        agent_inputs = self._build_inputs(ep_batch, t)
        bs = ep_batch.batch_size

        agent_inputs = agent_inputs.view(bs, self.n_agents, -1)

        h_states = self.hidden_states.view(bs, self.n_agents, -1)

        top_qs, micro_qs, macro_idxs, phis = [], [], [], []
        new_hidden_states = []


        for i in range(self.n_agents):
            tf = teacher_forcing_actions[:, i] if teacher_forcing_actions is not None else None

            tq, mq, midx, phi, new_h = self.agent(
                agent_inputs[:, i],
                h_states[:, i],
                teacher_forcing_actions=tf,
                epsilon=epsilon
            )

            top_qs.append(tq)
            micro_qs.append(mq)
            macro_idxs.append(midx)
            phis.append(phi)
            new_hidden_states.append(new_h)

        self.hidden_states = torch.stack(new_hidden_states, dim=1).view(bs * self.n_agents, -1)

        return {
            "top_q": torch.stack(top_qs, dim=1).squeeze(-1),  # [bs, n_agents]
            "micro_qs": micro_qs,
            "macro_idx": torch.stack(macro_idxs, dim=1),  # [bs, n_agents, num_components]
            "phi": torch.stack(phis, dim=1)  # [bs, n_agents, embed_dim]
        }
