import torch
import torch.nn as nn
import torchvision.models as models

class ResNetDQN(nn.Module):
    """
    ResNet-based Deep Q-Network (DQN) model for reinforcement learning.

    This model leverages a pretrained ResNet-18 architecture to process input states 
    and output Q-values for each possible action. The initial convolutional layer of 
    ResNet is adapted to accept single-channel input. The ResNet model extracts 
    high-level features from the input, and the fully connected layers map these 
    features to action values.

    Attributes:
        resnet (nn.Module): Pretrained ResNet-18 model with modified input layer.
        fc (nn.Sequential): Fully connected layers with ReLU activations for 
                            mapping ResNet features to action values.
    """
    def __init__(self, input_shape, n_actions):
        super(ResNetDQN, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        resnet_out = self.resnet(x)
        return self.fc(resnet_out)

