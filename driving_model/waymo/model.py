import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
import albumentations as A
import numpy as np

class TrajectoryModel(nn.Module):
    def __init__(self):
        super().__init__()