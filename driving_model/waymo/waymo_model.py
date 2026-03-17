import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

class DrivingModel(nn.Module):
    def __init__(self, num_waypoints=20):
        super().__init__()
        
        self.num_waypoints = num_waypoints

        self.backbone = models.resnet18(weights="IMAGENET1K_V1")

        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_waypoints * 2)
        )

    def forward(self, x):
        x = self.backbone(x)

        x = x.view(-1, self.num_waypoints, 2)

        return x