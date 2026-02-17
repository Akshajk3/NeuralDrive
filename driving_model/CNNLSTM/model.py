import torch
import torch.nn as nn
import torch.functional as F

class SteeringCNNLSTM(nn.Module):
    def __init__(self, seq_len=5, input_channels=3):
        super.__init__()
        self.seq_len = seq_len

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2)
        )