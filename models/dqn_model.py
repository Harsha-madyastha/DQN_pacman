import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model for reinforcement learning.

    This model uses convolutional neural networks to process input states and 
    output Q-values for each possible action. It consists of convolutional layers 
    followed by fully connected layers. The convolutional layers extract spatial 
    features from the input, while the fully connected layers map these features 
    to action values.

    Attributes:
        conv (nn.Sequential): Convolutional layers with ReLU activations.
        fc (nn.Sequential): Fully connected layers with ReLU activations.
    """
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_out(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
