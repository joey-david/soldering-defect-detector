import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import train2

# Load the trained model
model = train2.Autoencoder().cuda()
model.load_state_dict(torch.load('autoencoder_model.pth'))
model.eval()  # Set the model to evaluation mode

# Function to preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.cuda()  # Move to GPU if available

# Function to predict and display results
def predict_and_display(image_path, threshold=0.5):
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Get the model's reconstruction
    with torch.no_grad():
        reconstructed_image = model(image)
    
    # Compute reconstruction error
    reconstruction_error = ((image - reconstructed_image) ** 2).mean().item()
    
    # Classify as defective or normal
    if reconstruction_error > threshold:
        classification = "Defective"
    else:
        classification = "Normal"
    
    # Print classification and additional info
    print(f"Image: {image_path}")
    print(f"Classification: {classification}")
    print(f"Reconstruction Error: {reconstruction_error:.4f}")
    print(f"Threshold: {threshold}")
    print("----------------------------------------")
    
    # Visualize the original, reconstructed, and error map
    reconstruction_error_map = ((image - reconstructed_image) ** 2).mean(axis=1).squeeze().cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze().cpu().numpy().transpose(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    
    # Reconstructed Image
    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_image.squeeze().cpu().numpy().transpose(1, 2, 0))
    plt.title('Reconstructed Image')
    plt.axis('off')
    
    # Reconstruction Error Map
    plt.subplot(1, 3, 3)
    plt.imshow(reconstruction_error_map, cmap='jet')
    plt.title('Reconstruction Error Map')
    plt.axis('off')
    
    plt.suptitle(f'Classification: {classification} (Error: {reconstruction_error:.4f})')
    plt.show()

# Select an image and get its classification
image_path = 'path_to_your_image.png'  # Replace with the path to your image
predict_and_display(image_path, threshold=0.5)  # Adjust threshold as needed