import torch
import torch.nn as nn
from torchvision import models
import joblib


def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    # Load config
    config = joblib.load(
        "D:/MedVision_Ai/backend/weights/brain_tumer/brain_mri_config.pkl"
    )

    # Create model architecture
    model = models.densenet121(
        weights=None
    )

    model.classifier = nn.Linear(
        model.classifier.in_features,
        config["num_classes"]
    )

    # Load weights
    model.load_state_dict(
        torch.load(
            "D:/MedVision_Ai/backend/weights/brain_tumer/brain_mri_model.pt",
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    return model, config, device