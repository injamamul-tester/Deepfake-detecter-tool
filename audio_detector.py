"""
Audio Deepfake Detection Module
Extracts MFCC/spectrogram, uses NN for synthetic voice detection
"""
import torch
import librosa
import numpy as np
from models.load_audio_model import load_audio_model

def detect_audio_deepfake(audio_path):
    """
    Detects deepfake in an audio file.
    Returns (is_deepfake, confidence_score)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_audio_model(device)
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc, axis=1)
    mfcc = torch.tensor(mfcc).float().unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(mfcc)
        prob = torch.sigmoid(output).item()
    is_deepfake = prob > 0.5
    return is_deepfake, prob
