import numpy as np
import collections
import cv2
import torch

class MaxAndSkipEnv:
    """
    Wrapper for an environment that combines observations over multiple frames and skips actions.

    Attributes:
        env: The original environment to wrap.
        _skip (int): Number of frames to skip between actions.
        _obs_buffer (collections.deque): Circular buffer to store observations.
        legal_actions: List of legal actions in the environment.
    """
    def __init__(self, env, skip=16):
        self.env = env
        self._skip = skip
        self._obs_buffer = collections.deque(maxlen=skip)
        self.legal_actions = env.legal_actions

    def reset(self):
        self._obs_buffer.clear()
        observation = self.env.reset()
        self._obs_buffer.append(observation)
        return observation

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)

class ProcessFrame84Gray:
    """
    Preprocesses frames from an environment to grayscale and resize to 84x84.

    Attributes:
        env: The environment to wrap.
        legal_actions: List of legal actions in the environment.
    """
    def __init__(self, env):
        self.env = env
        self.legal_actions = env.legal_actions

    def reset(self):
        return self.process(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.process(obs), reward, done, info
    
    def __getattr__(self, name):
        return getattr(self.env, name)

    @staticmethod
    def process(frame):
        if len(frame.shape) > 2 and frame.shape[2] == 3:
            img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        elif len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            img_gray = frame
        else:
            raise ValueError("Unsupported input frame format")
        resized_screen = cv2.resize(img_gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized_screen.astype(np.uint8)
    

class BufferWrapper:
    """
    Wrapper to buffer observations over multiple time steps for RL algorithms.

    Attributes:
        env: The environment to wrap.
        dtype: Data type to use for the buffer.
        n_steps (int): Number of steps to buffer.
        legal_actions: List of legal actions in the environment.
        buffer (numpy.ndarray or None): Buffer to store observations.
    """
    def __init__(self, env, n_steps, legal_actions, dtype=np.float32):
        self.env = env
        self.dtype = dtype
        self.n_steps = n_steps
        self.legal_actions = legal_actions
        self.buffer = None  # Initialize buffer as None initially

    def reset(self):
        state = self.env.reset()
        self.buffer = np.zeros((self.n_steps, *state.shape), dtype=self.dtype)  # Initialize buffer with correct shape
        self.buffer[0] = state  # Assign initial state to buffer
        return self.observation()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.buffer[:-1] = self.buffer[1:]  # Shift buffer
        self.buffer[-1] = state  # Append new state to buffer
        return self.observation(), reward, done, info

    def observation(self):
        # Convert buffer to a 4D tensor
        return torch.tensor(self.buffer, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    def __getattr__(self, name):
        return getattr(self.env, name)