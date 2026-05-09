import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class AttentionBottleneckEncoder(nn.Module):
    def __init__(self, entity_feat_dim, embed_dim, num_heads=4):
        super(AttentionBottleneckEncoder, self).__init__()
        self.entity_proj = nn.Linear(entity_feat_dim, embed_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, entities, pad_mask=None):
        x = F.relu(self.entity_proj(entities))
        query, key, value = x[:, 0:1, :], x, x
        attn_out, _ = self.mhsa(query, key, value, key_padding_mask=pad_mask)
        out = self.layer_norm(query + attn_out)
        out = out + self.ffn(out)
        return out.squeeze(1)


class SequentialActionDecoder(nn.Module):
    def __init__(self, repr_dim, action_dims_list, hidden_dim=128):
        super(SequentialActionDecoder, self).__init__()
        self.action_dims_list = action_dims_list
        self.num_components = len(action_dims_list)
        self.total_action_dim = sum(action_dims_list)
        self.top_macro_q_net = nn.Sequential(
            nn.Linear(repr_dim + self.total_action_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.gru_context = nn.GRUCell(input_size=self.total_action_dim, hidden_size=hidden_dim)
        self.micro_q_heads = nn.ModuleList(
            [nn.Linear(repr_dim + hidden_dim, action_dim) for action_dim in action_dims_list])

    def forward(self, phi, teacher_forcing_actions=None, epsilon=0.0):
        if isinstance(phi, tuple): phi = phi[0]
        b, device = phi.shape[0], phi.device

        micro_q_outputs, macro_action_components, macro_action_indices = [], [], []
        h_t = torch.zeros(b, self.gru_context.hidden_size, device=device)
        a_t_prev = torch.zeros(b, self.total_action_dim, device=device)

        for step in range(self.num_components):
            h_t = self.gru_context(a_t_prev, h_t)
            q_micro = self.micro_q_heads[step](torch.cat([phi, h_t], dim=-1))
            micro_q_outputs.append(q_micro)

            if teacher_forcing_actions is not None:
                chosen_idx = teacher_forcing_actions[:, step].long()
            else:
                if random.random() < epsilon:
                    chosen_idx = torch.randint(0, self.action_dims_list[step], (b,), device=device)
                else:
                    chosen_idx = q_micro.argmax(dim=-1)

            macro_action_indices.append(chosen_idx.unsqueeze(1))
            a_t_onehot = F.one_hot(chosen_idx, num_classes=self.action_dims_list[step]).float()
            macro_action_components.append(a_t_onehot)

            a_t_prev = torch.zeros(b, self.total_action_dim, device=device)
            start_idx = sum(self.action_dims_list[:step])
            a_t_prev[:, start_idx: start_idx + self.action_dims_list[step]] = a_t_onehot

        macro_action_tensor = torch.cat(macro_action_components, dim=-1)
        macro_idx_tensor = torch.cat(macro_action_indices, dim=-1)
        top_q = self.top_macro_q_net(torch.cat([phi, macro_action_tensor], dim=-1))

        return micro_q_outputs, macro_action_tensor, macro_idx_tensor, top_q


class ABIMAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ABIMAgent, self).__init__()
        self.args = args
        self.encoder = AttentionBottleneckEncoder(args.entity_feat_dim, args.embed_dim, args.attn_heads)
        self.decoder = SequentialActionDecoder(args.embed_dim, args.action_dims_list, args.rnn_hidden_dim)

    def init_hidden(self):
        return self.encoder.entity_proj.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, teacher_forcing_actions=None, epsilon=0.0):
        b = inputs.shape[0]
        obs_dim = self.args.num_entities * self.args.entity_feat_dim
        entities = inputs[:, :obs_dim].reshape(b, self.args.num_entities, self.args.entity_feat_dim)

        phi_t = self.encoder(entities)
        mq, mt, midx, tq = self.decoder(phi_t, teacher_forcing_actions, epsilon)

        return tq, mq, midx, phi_t, hidden_state
