from sklearn.utils import shuffle
import pandas as pd
import os
import ntpath
import numpy as np
import matplotlib.pyplot as plt
from CNNLSTM import SteeringDataset, SteeringCNNLSTM
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

data_dir = '../sim_data'
columns = ['left', 'center', 'right', 'mask', 'steering']
data = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'), names=columns)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data['mask'] = data['mask'].apply(path_leaf)

data['steering'] = pd.to_numeric(data['steering'], errors='coerce')
data = data.dropna(subset=['steering'])

num_bins = 25
samples_per_bin = 200

hist, bins = np.histogram(data['steering'], num_bins)

center = (bins[:-1] + bins[1:]) * 0.5
plt.bar(center, hist, width=0.025)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

remove_list = []
for j in range(num_bins):
    list_ = []
    for i in range(len(data['steering'])):
        if data['steering'].iloc[i] >= bins[j] and data['steering'].iloc[i] <= bins[j+1]:
            list_.append(i)
    list_ = shuffle(list_)
    list_ = list_[samples_per_bin:]
    remove_list.extend(list_)

print(f'removed {len(remove_list)}')
data.drop(data.index[remove_list], inplace=True)
print(f'remaining data {len(data)}')

hist, _ = np.histogram(data['steering'], (num_bins))
plt.bar(center, hist, width=0.025)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()


def load_img_steering(rgb_dir, mask_dir, df):
    rgb_path = []
    mask_path = []
    steering = []

    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        mask = indexed_data[3]

        rgb_path.append(os.path.join(rgb_dir, center.strip()))
        steering.append(float(indexed_data[4]))
        mask_path.append(os.path.join(mask_dir, mask.strip()))

        rgb_path.append(os.path.join(rgb_dir, left.strip()))
        steering.append(float(indexed_data[4] + 0.15))
        mask_path.append(os.path.join(mask_dir, mask.strip()))

        rgb_path.append(os.path.join(rgb_dir, right.strip()))
        steering.append(float(indexed_data[4] - 0.15))
        mask_path.append(os.path.join(mask_dir, mask.strip()))
    
    rgb_paths = np.asarray(rgb_path)
    steerings = np.asarray(steering)
    mask_paths = np.asarray(mask_path)

    return rgb_paths, steerings, mask_paths

rgb_paths, steerings, mask_paths = load_img_steering(data_dir + '/RGB', data_dir + '/masks', data)

split_index = int(0.8 * len(rgb_paths))

rgb_train = rgb_paths[:split_index]
mask_train = mask_paths[:split_index]
steer_train = steerings[:split_index]

rgb_valid = rgb_paths[split_index:]
mask_valid = mask_paths[split_index:]
steer_valid = steerings[split_index:]

train_dataset = SteeringDataset(
    rgb_train,
    mask_train,
    steer_train,
    seq_len=5,
    training=True
)

val_dataset = SteeringDataset(
    rgb_valid,
    mask_valid,
    steer_valid,
    seq_len=5,
    training=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

valid_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Device: {device}")

model = SteeringCNNLSTM(seq_len=5).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 15

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for rgb_seq, mask_seq, steering in train_loader:
        rgb_seq = rgb_seq.to(device, non_blocking=True)
        mask_seq = mask_seq.to(device, non_blocking=True)
        steering = steering.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(rgb_seq, mask_seq)
        loss = criterion(outputs, steering)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_loss += loss.item() * rgb_seq.size(0)
    
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for rgb_seq, mask_seq, steering in valid_loader:
            rgb_seq = rgb_seq.to(device, non_blocking=True)
            mask_seq = mask_seq.to(device, non_blocking=True)
            steering = steering.to(device, non_blocking=True)

            outputs = model(rgb_seq, mask_seq)
            loss = criterion(outputs, steering)

            val_loss += loss.item() * rgb_seq.size(0)
        
    val_loss /= len(valid_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}")
    
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
    
torch.save(model.state_dict(), 'steering_cnn_lstm.pth')