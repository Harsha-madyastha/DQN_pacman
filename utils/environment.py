from ale_py import ALEInterface
import numpy as np
import cv2

class CustomMsPacmanEnv:
    """
    Custom environment wrapper for Ms. Pacman using ALEInterface.

    Attributes:
        ale: ALEInterface object for interfacing with the Atari Learning Environment.
        legal_actions: List of legal actions in the environment.
        observation_space: Shape of the observation space (screen dimensions).
        action_space: Number of possible actions in the environment.
    """
    def __init__(self, rom_path):
        self.ale = ALEInterface()
        self.ale.loadROM(rom_path)
        self.legal_actions = self.ale.getLegalActionSet()
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.reset()

    def reset(self):
        self.ale.reset_game()
        return self.get_screen()

    def step(self, action):
        reward = self.ale.act(action)
        screen_obs = self.get_screen()
        done = self.ale.game_over()
        return screen_obs, reward, done, None

    def get_screen(self):
        screen = self.ale.getScreenRGB()
        return screen
    
    def _get_observation_space(self):
        screen_shape = self.get_screen().shape
        return np.array(screen_shape)

    def _get_action_space(self):
        return len(self.legal_actions)
