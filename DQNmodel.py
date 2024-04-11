import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self, inp_dim, action_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(inp_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)  # Directly output action Q-values
        )
        self.epsilon = 1  # Exploration rate

    def forward(self, x):
        return self.layers(x / 8)  # Preprocessing input and passing through the network

    def act(self, state, mask):
        if random.random() > self.epsilon:  # Epsilon-greedy action selection
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state)
            masked_q_values = q_values.masked_fill(torch.FloatTensor(mask).unsqueeze(0) == 0, float('-inf'))
            action = masked_q_values.max(1)[1].data[0].item()
        else:
            indices = np.nonzero(mask)[0]
            action = random.choice(indices)  # Choose a random valid action
        return action

class Buffer():
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, mask, reward, new_state, new_mask, terminal):
        self.buffer.append((state, action, mask, reward, new_state, new_mask, terminal))

    def sample(self, batch_size):
        states, actions, masks, rewards, new_states, new_mask, terminals = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(masks), np.array(rewards), np.array(new_states), np.array(new_mask), np.array(terminals)