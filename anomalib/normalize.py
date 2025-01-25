import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# Paths
DATASET_PATH = "dataset/"
MODEL_PATH = "solder_defect_classifier.pth"
CLASS_NAMES = ["SL", "ST_Inf", "STP", "ST_Sup", "ST_Sup_Pli", "Sans_Defaut"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),         # Convert image to PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Load Train and Validation Datasets
train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "train"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
