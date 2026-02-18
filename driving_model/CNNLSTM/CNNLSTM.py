import os
import cv2
import random
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
import albumentations as A
import numpy as np

class SteeringDataset(Dataset):
    def __init__(self, rgb_paths, mask_paths, steerings, seq_len=5, training=True):
        self.rgb_paths = rgb_paths
        self.mask_paths = mask_paths
        self.steerings = steerings
        self.seq_len = seq_len
        self.training = training
    
    def __len__(self):
        return len(self.mask_paths) - self.seq_len + 1
    
    def __getitem__(self, idx):
        rgb_seq = []
        mask_seq = []
        steering_seq = []

        if self.training:
            do_pan = np.random.rand() < 0.5
            do_zoom = np.random.rand() < 0.5
            do_brightness = np.random.rand() < 0.5
            do_flip = np.random.rand() < 0.5
        else:
            do_pan = do_zoom = do_brightness = do_flip = False

        for i in range(self.seq_len):
            rgb = cv2.imread(self.rgb_paths[idx + i])
            mask = cv2.imread(self.mask_paths[idx + i], cv2.IMREAD_GRAYSCALE)
            steering_angle = self.steerings[idx + i]

            if do_pan:
                rgb, mask = self.pan(rgb, mask)
            if do_zoom:
                rgb, mask = self.zoom(rgb, mask)
            if do_brightness:
                rgb = self.img_random_brightness(rgb)
            if do_flip:
                rgb, mask, steering_angle = self.img_random_flip(rgb, mask, steering_angle)

            rgb, mask = self.preprocess(rgb, mask)

            rgb_seq.append(rgb)
            mask_seq.append(mask)
            steering_seq.append(steering_angle)

        rgb_seq = np.stack(rgb_seq)
        mask_seq = np.stack(mask_seq)
        steering_seq = np.stack(steering_seq)

        return torch.tensor(rgb_seq, dtype=torch.float32), torch.tensor(mask_seq, dtype=torch.float32), torch.tensor(steering_seq[-1], dtype=torch.float32)

    def preprocess(self, img, mask):
        img = img[90:, :, :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.resize(img, (200, 66))
        img = img / 255
        img = np.transpose(img, (2, 0, 1))

        mask = mask[90:, :]
        mask = cv2.resize(mask, (200, 66), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)

        return img, mask

    def pan(self, img, mask):
        pan = A.Compose(
            [A.Affine(translate_percent={"x" : (-0.1, 0.1), "y" : (-0.1, 0.1)})],
            additional_targets={"mask" : "mask"}
        )
        augmented = pan(image=img, mask=mask)
        return augmented["image"], augmented["mask"]
    
    def zoom(self, img, mask):
        zoom = A.Compose(
            [A.Affine(scale=(1.0, 1.3))],
            additional_targets={"mask" : "mask"}
        )
        augmented = zoom(image=img, mask=mask)
        return augmented["image"], augmented["mask"]

    def img_random_brightness(self, img):
        brightness = A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.0,
        p=1.0)

        augmented = brightness(image=img)
        return augmented["image"]
    
    def img_random_flip(self, img, mask, steering_angle):
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)
        steering_angle = -steering_angle
        return img, mask, steering_angle

class SteeringCNNLSTM(nn.Module):
    def __init__(self, seq_len=5, input_channels=3):
        super().__init__()
        self.seq_len = seq_len

        self.rgb_cnn = nn.Sequential(
            nn.Conv2d(input_channels, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=3),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ELU(),
            nn.Flatten()
        )

        self.mask_cnn = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=3),
            nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3),
            nn.ELU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_rgb = torch.zeros(1, 3, 66, 200)
            dummy_mask = torch.zeros(1, 1, 66, 200)
            rgb_feat_size = self.rgb_cnn(dummy_rgb).shape[1]
            mask_feat_size = self.mask_cnn(dummy_mask).shape[1]
            self.cnn_out_size = rgb_feat_size + mask_feat_size
        
        self.lstm = nn.LSTM(input_size=self.cnn_out_size, hidden_size=100, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )

    def forward(self, rgb_seq, mask_seq):
        batch_size, seq_len, C_rgb, H, W = rgb_seq.shape
        _, _, C_mask, _, _ = mask_seq.shape

        rgb_seq = rgb_seq.view(batch_size * seq_len, C_rgb, H, W)
        mask_seq = mask_seq.view(batch_size * seq_len, C_mask, H, W)

        rgb_feat = self.rgb_cnn(rgb_seq)
        mask_feat = self.mask_cnn(mask_seq)

        combined = torch.cat([rgb_feat, mask_feat], dim=1)
        combined = combined.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(combined)
        out = self.fc(lstm_out[:, -1, :])
        return torch.tanh(out.squeeze(1))