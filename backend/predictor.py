
from PIL import Image
import torch
from torchvision import transforms

from model_loader import load_model


# Load model once when the backend starts
model, config, device = load_model()


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize(
        (config["image_size"], config["image_size"])
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=config["mean"],
        std=config["std"]
    )
])


def predict(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Preprocess image
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Model prediction
    with torch.no_grad():
        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    # Get predicted class
    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = config["classes"][predicted_index]

    confidence = (
        probabilities[predicted_index].item() * 100
    )

    # Get all class probabilities
    all_probs = {}

    for class_name, prob in zip(
        config["classes"],
        probabilities
    ):
        all_probs[class_name] = round(
            prob.item() * 100, 2
        )

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "probabilities": all_probs
    }