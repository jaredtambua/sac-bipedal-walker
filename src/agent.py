from pathlib import Path

import numpy as np
import torch

from src.sac import SAC
from src.device import device


class Agent:
    def __init__(
        self,
        env,
        seed=42,
        discount=0.99,
        tau=0.005,
        buffer_size=1000000,
        batch_size=256,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        initial_alpha=0.2,
        warmup_steps=10000,
        updates_per_step=1,
        hidden=256,
    ):
        self.env = env

        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.algo = SAC(
            observation_dim=observation_dim,
            action_dim=action_dim,
            seed=seed,
            discount=discount,
            tau=tau,
            buffer_size=buffer_size,
            batch_size=batch_size,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            initial_alpha=initial_alpha,
            hidden=hidden,
        )

        self.warmup_steps = int(warmup_steps)
        self.updates_per_step = int(updates_per_step)
        self.total_steps = 0

    def _safe_action(self, action):
        action = np.asarray(action, dtype=np.float32)

        if action.shape != self.env.action_space.shape:
            raise ValueError(
                f"Bad action shape: got {action.shape}, "
                f"expected {self.env.action_space.shape}"
            )

        if not np.isfinite(action).all():
            raise ValueError(f"Action contains NaN/inf: {action}")

        return np.clip(
            action,
            self.env.action_space.low.astype(np.float32),
            self.env.action_space.high.astype(np.float32),
        ).astype(np.float32)

    def episode(self):
        episode_return = 0.0
        episode_length = 0

        observation, info = self.env.reset()
        done = False

        while not done:
            if self.total_steps < self.warmup_steps:
                action = self.env.action_space.sample().astype(np.float32)
            else:
                action = self.algo.sample_action(
                    observation,
                    explore=True,
                )

            action = self._safe_action(action)

            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            # Clip large negative rewards for training stability.
            reward = max(reward, -5.0)

            episode_return += float(reward)
            episode_length += 1

            done = bool(terminated or truncated)

            self.algo.add(
                observation,
                action,
                reward,
                next_observation,
                done,
            )

            if self.total_steps >= self.warmup_steps:
                for _ in range(self.updates_per_step):
                    self.algo.learn()

            observation = next_observation
            self.total_steps += 1

        return {
            "episode_return": episode_return,
            "episode_length": episode_length,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "actor": self.algo.actor.state_dict(),
                "critic1": self.algo.critic1.state_dict(),
                "critic2": self.algo.critic2.state_dict(),
                "target_critic1": self.algo.target_critic1.state_dict(),
                "target_critic2": self.algo.target_critic2.state_dict(),
                "log_alpha": self.algo.log_alpha.detach().cpu(),
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=device)

        self.algo.actor.load_state_dict(checkpoint["actor"])
        self.algo.critic1.load_state_dict(checkpoint["critic1"])
        self.algo.critic2.load_state_dict(checkpoint["critic2"])
        self.algo.target_critic1.load_state_dict(checkpoint["target_critic1"])
        self.algo.target_critic2.load_state_dict(checkpoint["target_critic2"])

        self.algo.log_alpha.data = checkpoint["log_alpha"].to(device)
        self.total_steps = checkpoint.get("total_steps", 0)
