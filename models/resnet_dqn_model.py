import torch
import torch.nn as nn
import torchvision.models as models

class ResNetDQN(nn.Module):
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
    
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.resnet = models.resnet18(pretrained=True)
        print(self.resnet)

        # Freeze ResNet18 layers
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        batch_size = x.size(0)
        resnet_out = self.resnet(x)
        resnet_out = resnet_out.view(batch_size, -1)
        return self.fc(resnet_out)
