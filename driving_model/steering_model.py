import torch
import cv2
import numpy as np
from CNNLSTM.CNNLSTM import SteeringCNNLSTM

class SteeringModel:
    def __init__(self, model_path="driving_model/CNNLSTM/steering_cnn_lstm.pth"):
        self.model_path = model_path
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        
        self.model = SteeringCNNLSTM(seq_len=5).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        self.seq_len = 5
        self.rgb_buffer = []
        self.mask_buffer = []
    
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
    
    def model_predict(self, img, mask):
        self.rgb_buffer.append(img)
        self.mask_buffer.append(mask)

        if len(self.rgb_buffer) > self.seq_len:
            self.rgb_buffer.pop(0)
            self.mask_buffer.pop(0)
        
        if len(self.rgb_buffer) < self.seq_len:
            return 0.0

        rgb_seq = np.stack(self.rgb_buffer, axis=0)
        mask_seq = np.stack(self.mask_buffer, axis=0)

        rgb_seq = torch.tensor(rgb_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_seq = torch.tensor(mask_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            steering = self.model(rgb_seq, mask_seq)
            steering = steering.item()
            steering = np.clip(steering, -1.0, 1.0)
        
        return steering