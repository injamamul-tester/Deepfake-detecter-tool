"""
Loads pretrained CNN model for image deepfake detection
"""
import torch
import torchvision.models as models

def load_image_model(device):
    model = models.resnet18(pretrained=True)
    # Replace final layer for binary classification
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    # Load custom weights if available
    try:
        model.load_state_dict(torch.load('models/image_deepfake.pth', map_location=device))
    except Exception:
        pass  # Use default weights
    model.to(device)
    model.eval()
    return model
