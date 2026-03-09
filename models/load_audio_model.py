"""
Loads simple NN model for audio deepfake detection
"""
import torch
import torch.nn as nn

def load_audio_model(device):
    class AudioNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(40, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 1)
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    model = AudioNet()
    try:
        model.load_state_dict(torch.load('models/audio_deepfake.pth', map_location=device))
    except Exception:
        pass  # Use default weights
    model.to(device)
    model.eval()
    return model
