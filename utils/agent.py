import numpy as np
import torch
from random import randrange
from collections import namedtuple

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'new_state'])

# Agent
class Agent:
    """
    Agent that interacts with an environment and collects experiences.

    Attributes:
        env: The environment where the agent interacts.
        exp_buffer: A buffer to store experiences (state, action, reward, done, new_state).
        state: Current state of the agent in the environment.
        total_reward: Cumulative reward collected by the agent in the current episode.
    """
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device='cpu'):
        """
        Take a step in the environment using an epsilon-greedy strategy.

        Args:
            net: Neural network model to compute Q-values.
            epsilon (float): Probability of taking a random action.

        Returns:
            done_reward: Total reward collected if the episode is done, otherwise None.
        """
        done_reward = None

        if np.random.random() < epsilon:
            action = randrange(len(self.env.legal_actions))
        else:
            state_a = np.array(self.state)
            state_v = torch.tensor(state_a).to(device, dtype=torch.float32)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward