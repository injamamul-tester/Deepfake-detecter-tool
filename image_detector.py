"""
Image Deepfake Detection Module
Uses pretrained CNN (ResNet/EfficientNet) for face deepfake detection
"""
import torch
import cv2
import numpy as np
from models.load_image_model import load_image_model
from utils.face_utils import detect_face

def detect_image_deepfake(image_path):
    """
    Detects deepfake in an image file.
    Returns (is_deepfake, confidence_score)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_image_model(device)
    img = cv2.imread(image_path)
    face = detect_face(img)
    if face is None:
        return False, 0.0  # No face detected
    # Preprocess face for model
    face = cv2.resize(face, (224, 224))
    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))
    face = torch.tensor(face).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(face)
        prob = torch.sigmoid(output).item()
    is_deepfake = prob > 0.5
    return is_deepfake, prob
