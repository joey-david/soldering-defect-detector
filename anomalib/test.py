import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import torchvision

def load_model(model_path):
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict(model, image, class_names, device):
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Example Usage
if __name__ == "__main__":
    test_image_path = "dataset/Defaut/SL/image_101_SL.png"

    model = load_model(MODEL_PATH).to(DEVICE)
    image = preprocess_image(test_image_path, transform)
    predicted_class = predict(model, image, CLASS_NAMES, DEVICE)

    print(f"Predicted Class: {predicted_class}")
