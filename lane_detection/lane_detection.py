from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
import json
import cv2
import random

path = kagglehub.dataset_download("manideep1108/tusimple")
print("Path to dataset files: ", path)

class TUSimpleDataset(Dataset):
    def __init__(self, base_path, json_file, image_transform=None, mask_transform=None):
        self.base_path = base_path
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        with open(json_file, 'r') as f:
            self.annotations = [json.loads(line) for line in f]

            self.annotations = [ann for ann in self.annotations if 'lanes' in ann and len(ann['lanes']) > 0]
    
    def __len__(self):
        return len(self.annotations)
    
    def create_lane_mask(self, annotation, img_height=720, img_width=1280):
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        lanes = annotation['lanes']
        h_samples = annotation['h_samples']

        for lane in lanes:
            if len(lane) > 0:
                lane_points = []
                for i, x in enumerate(lane):
                    if x >= 0:
                        y = h_samples[i]
                        lane_points.append([x, y])

                
                if len(lane_points) > 1:
                    lane_points = np.array(lane_points, dtype=np.int32)
                    cv2.polylines(mask, [lane_points], False, 255, thickness=6)

        return mask
    
    def __getitem__(self, index):
        annotation = self.annotations[index]

        image_path = os.path.join(self.base_path, annotation['raw_file'])

        image = Image.open(image_path).convert("RGB")

        mask = self.create_lane_mask(annotation)
        mask_img = Image.fromarray(mask).convert("L")

        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_img = mask_img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            mask_img = self.mask_transform(mask_img)

        return image, mask_img
    
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 512)

        self.up2 = nn.ConvTranspose2d(512, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        up2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        return self.final(dec1)
    
def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def combined_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img_tensor * std + mean
    
def train(model, train_loader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Train Epoch: {epoch + 1}, Batch {batch_idx}, Loss {loss.item():.4f}")
    
    return total_loss / len(train_loader)

def test(model, test_loader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    base_path = os.path.join(path, "TUSimple", "train_set")
    json_file = os.path.join(base_path, "label_data_0313.json")

    print(f"Loading dataset from: {json_file}")
    print(f"Base path: {base_path}")

    image_transform = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 512), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    try:
        dataset = TUSimpleDataset(base_path, json_file, image_transform, mask_transform)
        print(f"Dataset successfully loaded with {len(dataset)} samples")

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

        model = UNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = combined_loss

        for epoch in range(20):
            print(f"\nEpoch: {epoch+1}")
            train_loss = train(model, train_loader, optimizer, loss_fn, device, epoch)
            test_loss = test(model, val_loader, loss_fn, device)
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        torch.save(model.state_dict(), 'model.h5')
        print("Model saved as model.h5")

        model.eval()

        sample_img, sample_mask = dataset[15]

        with torch.no_grad():
            input_tensor = sample_img.unsqueeze(0).to(device)
            pred = torch.sigmoid(model(input_tensor))
            pred_mask = pred.squeeze().cpu().numpy() > 0.5
        
        pred_mask_img = Image.fromarray((pred_mask * 255).astype('uint8'))
        
        sample_img_denorm = denormalize(sample_img.cpu())
        sample_img_np = sample_img_denorm.permute(1, 2, 0).numpy()
        sample_img_np = np.clip(sample_img_np * 255, 0, 255).astype(np.uint8)
        sample_image_pil = Image.fromarray(sample_img_np)

        H, W = pred_mask.shape

        red_overlay = Image.new("RGB", (W, H), (255, 0, 0))
        mask_overlay = Image.composite(red_overlay, sample_image_pil, pred_mask_img)
        blended = Image.blend(sample_image_pil, mask_overlay, alpha=0.5)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(sample_image_pil)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(sample_mask.squeeze().numpy(), cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(blended)
        plt.title("Predicted Lane Overlay")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check the dataset paths and structure")