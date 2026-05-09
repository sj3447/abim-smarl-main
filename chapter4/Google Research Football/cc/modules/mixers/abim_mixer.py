import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ABIMMixer(nn.Module):
    def __init__(self, args):
        super(ABIMMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.embed_dim
        self.hypernet_hidden = args.hypernet_hidden

        self.hyper_feat_extractor = nn.Sequential(
            nn.Linear(self.state_dim, self.hypernet_hidden),
            nn.ReLU(),
            nn.Linear(self.hypernet_hidden, self.hypernet_hidden),
            nn.ReLU()
        )

        self.hyper_w_1 = nn.Linear(self.hypernet_hidden, self.embed_dim * self.n_agents)
        self.hyper_b_1 = nn.Linear(self.hypernet_hidden, self.embed_dim)
        self.hyper_w_2 = nn.Linear(self.hypernet_hidden, self.embed_dim)
        self.hyper_b_2 = nn.Sequential(
            nn.Linear(self.hypernet_hidden, self.hypernet_hidden),
            nn.ReLU(),
            nn.Linear(self.hypernet_hidden, 1)
        )

        self.adv_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hypernet_hidden),
            nn.ReLU(),
            nn.Linear(self.hypernet_hidden, self.n_agents)
        )

    def forward(self, agent_qs, states):
        bs, t = agent_qs.size(0), agent_qs.size(1)

        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        hyper_in = self.hyper_feat_extractor(states)

        w1 = torch.abs(self.hyper_w_1(hyper_in)).reshape(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(hyper_in).reshape(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w2 = torch.abs(self.hyper_w_2(hyper_in)).reshape(-1, self.embed_dim, 1)
        b2 = self.hyper_b_2(hyper_in).reshape(-1, 1, 1)
        q_tot_monotonic = torch.bmm(hidden, w2) + b2

        adv_weights = self.adv_net(states).reshape(-1, 1, self.n_agents)
        adv_bypass = torch.bmm(agent_qs, adv_weights.transpose(1, 2))

        q_tot = q_tot_monotonic + adv_bypass

        return q_tot.reshape(bs, t, 1)
