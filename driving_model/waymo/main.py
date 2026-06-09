import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py
from torchvision import models
from torchvision import transforms


class WaymoDataset:
    def __init__(self, h5_path):
        self.path = h5_path
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),

        ])

        with h5py.File(h5_path, 'r') as f:
            self.frame_keys = sorted(f['frames'])
        
    def __len__(self):
        return len(self.frame_keys)

    def __getitem__(self, idx):
        key = self.frame_keys[idx]

        with h5py.File(self.path, 'r') as f:
            frame = f['frames'][key]

            img_center = frame['images/center'][:]
            img_left = frame['images/left'][:]
            img_right = frame['images/right'][:]
        
            pose = frame['pose'][:]
            
            waypoints_2d = frame['waypoints_2d'][:]
            waypoints_3d = frame['waypoints_3d'][:]

            center_intrinsic  = frame['calibration/center/intrinsic'][:]
            center_extrinsic  = frame['calibration/center/extrinsic'][:]

        img_center = torch.tensor(img_center).permute(2, 0, 1).float() / 255.0
        img_left = torch.tensor(img_left).permute(2, 0, 1).float() / 255.0
        img_right = torch.tensor(img_right).permute(2, 0, 1).float() / 255.0

        pose = torch.tensor(pose).float()

        waypoints_2d = torch.tensor(waypoints_2d).float()
        waypoints_3d = torch.tensor(waypoints_3d).float()

        center_intrinsic = torch.tensor(center_intrinsic).float()
        center_extrinsic = torch.tensor(center_extrinsic).float()

        return {
            'img_center' : img_center,
            'img_left' : img_left,
            'img_right' : img_right,
            'pose' : pose,
            'waypoints_3d' : waypoints_3d
        }
    
class WaymoModel(nn.Module):
    def __init__(self):
        pass

dataset = WaymoDataset('waymo_open_dataset/training.h5')
print(dataset.__len__())
# item = dataset.__getitem__(0)
# img_center = item['img_center']

# plt.imshow(img_center)
# plt.axis('off')
# plt.show()