import collections

import numpy as np
import torch

device = torch.device("cpu")


class ReplayBuffer:
    def __init__(self, capacity=1000000, seed=42):
        self.buffer = collections.deque(maxlen=int(capacity))
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        transition = (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.float32(reward),
            np.array(next_state, dtype=np.float32),
            np.float32(done),
        )
        self.buffer.append(transition)

    def sample(self, batch_size):
        indices = self.rng.choice(
            len(self.buffer),
            size=batch_size,
            replace=False,
        )

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=device)
        rewards = torch.tensor(
            np.array(rewards), dtype=torch.float32, device=device
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=device
        )
        dones = torch.tensor(
            np.array(dones), dtype=torch.float32, device=device
        ).unsqueeze(1)

        return states, actions, rewards, next_states, dones
