"""
Loads pretrained CNN model for image deepfake detection
"""
import torch
import torchvision.models as models

def load_image_model(device):
    # Use EfficientNet for improved accuracy
    model = models.efficientnet_b0(pretrained=True)
    # Replace classifier for binary classification
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, 1)
    # Load custom weights if available
    try:
        model.load_state_dict(torch.load('models/image_deepfake.pth', map_location=device))
    except Exception:
        pass  # Use default weights
    model.to(device)
    model.eval()
    return model
