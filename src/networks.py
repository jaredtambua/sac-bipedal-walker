import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianPolicy(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden=256):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden, action_dim)
        self.log_std_layer = nn.Linear(hidden, action_dim)

    def forward(self, observations):
        features = self.trunk(observations)

        mean = self.mean_layer(features)

        log_std = self.log_std_layer(features)
        log_std = torch.clamp(
            log_std,
            min=LOG_STD_MIN,
            max=LOG_STD_MAX,
        )

        return mean, log_std

    def sample(self, observations):
        mean, log_std = self(observations)

        std = log_std.exp()

        distribution = Normal(mean, std)

        pre_tanh = distribution.rsample()

        action = torch.tanh(pre_tanh)

        log_prob = distribution.log_prob(pre_tanh)

        log_prob -= torch.log(1 - action.pow(2) + 1e-6)

        log_prob = log_prob.sum(
            dim=1,
            keepdim=True,
        )

        return action, log_prob

    @torch.no_grad()
    def act(self, observations):
        mean, _ = self(observations)
        return torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        hidden=256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                observation_dim + action_dim,
                hidden,
            ),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        observations,
        actions,
    ):
        x = torch.cat(
            [observations, actions],
            dim=1,
        )

        return self.network(x)
