import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from src.networks import GaussianPolicy, QNetwork
from src.replay_buffer import ReplayBuffer
from src.device import device


@torch.no_grad()
def soft_update(target_network, source_network, tau):
    for target_param, source_param in zip(
        target_network.parameters(),
        source_network.parameters(),
    ):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * source_param.data)


class SAC:
    def __init__(
        self,
        observation_dim,
        action_dim,
        seed=42,
        discount=0.99,
        tau=0.005,
        buffer_size=1000000,
        batch_size=256,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        initial_alpha=0.2,
        target_entropy=None,
        hidden=256,
    ):
        self.discount = float(discount)
        self.tau = float(tau)
        self.batch_size = int(batch_size)

        self.actor = GaussianPolicy(
            observation_dim,
            action_dim,
            hidden=hidden,
        ).to(device)

        self.critic1 = QNetwork(
            observation_dim,
            action_dim,
            hidden=hidden,
        ).to(device)

        self.critic2 = QNetwork(
            observation_dim,
            action_dim,
            hidden=hidden,
        ).to(device)

        self.target_critic1 = deepcopy(self.critic1).to(device)
        self.target_critic2 = deepcopy(self.critic2).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.log_alpha = torch.tensor(
            np.log(initial_alpha),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )

        self.alpha_optimizer = optim.Adam(
            [self.log_alpha],
            lr=alpha_lr,
        )

        if target_entropy is None:
            target_entropy = -float(action_dim)

        self.target_entropy = float(target_entropy)

        self.replay = ReplayBuffer(
            capacity=buffer_size,
            seed=seed,
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def add(self, state, action, reward, next_state, done):
        self.replay.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    @torch.no_grad()
    def sample_action(self, observation, explore=True):
        observation = torch.tensor(
            observation,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        if explore:
            action, _ = self.actor.sample(observation)
        else:
            action = self.actor.act(observation)

        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size
        )
        not_done = 1.0 - dones

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)

            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            target_q = rewards + self.discount * not_done * (
                target_q - self.alpha.detach() * next_log_probs
            )

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)

        critic1_loss = nn.functional.mse_loss(current_q1, target_q)
        critic2_loss = nn.functional.mse_loss(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        new_actions, log_probs = self.actor.sample(states)

        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        soft_update(self.target_critic1, self.critic1, self.tau)
        soft_update(self.target_critic2, self.critic2, self.tau)
