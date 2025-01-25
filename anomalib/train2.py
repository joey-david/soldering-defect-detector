import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import numpy as np
import time

def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

class CustomDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        self.images = []
        self.labels = []
        
        good_dir = os.path.join(root_dir, 'Sans_Defaut')
        for img_name in os.listdir(good_dir):
            if img_name.endswith('.png'):
                self.images.append(os.path.join(good_dir, img_name))
                self.labels.append(0)
        
        defaut_dir = os.path.join(root_dir, 'Defaut')
        for subdir in os.listdir(defaut_dir):
            subdir_path = os.path.join(defaut_dir, subdir)
            if os.path.isdir(subdir_path):
                for img_name in os.listdir(subdir_path):
                    if img_name.endswith('.png'):
                        self.images.append(os.path.join(subdir_path, img_name))
                        self.labels.append(1)
        print(f"Normal images: {self.labels.count(0)}")
        print(f"Defective images: {self.labels.count(1)}")
        
        if self.mode == 'train':
            self.labels = None
        
        print(f"Dataset loaded with {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        image = image.convert('RGB')  # Convert grayscale image to RGB format
        if self.transform:
            image = self.transform(image)
        if self.mode == 'train':
            return image
        else:
            return image, self.labels[idx]

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=50):
    loss_values = []
    for epoch in range(num_epochs):
        training_start = time.time()
        model.train()
        running_loss = 0.0
        for data in train_loader:
            data = data.cuda()
            output = model(data)
            loss = 100_000 * criterion(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        loss_values.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Time since last epoch: {time.time() - training_start:.2f} seconds')
        scheduler.step()
    return loss_values

def evaluate_model(model, test_loader):
    y_true = []
    y_pred = []
    y_score = []
    model.eval()
    with torch.no_grad():
        for data, label in test_loader:
            data = data.cuda()
            recon = model(data)
            y_score_batch = ((data - recon)**2).mean(axis=(1,2,3)).cpu().numpy()
            y_pred_batch = (y_score_batch >= 0.5).astype(int)
            y_true.extend(label.numpy().tolist())
            y_pred.extend(y_pred_batch.tolist())
            y_score.extend(y_score_batch.tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_score)

def plot_roc_curve(y_true, y_score, auc_roc_score, learning_rate, num_epochs):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"roc_curve_{learning_rate}_{num_epochs}.png", dpi=300)

def plot_confusion_matrix(y_true, y_pred, learning_rate, num_epochs):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Defective'])
    disp.plot()
    plt.title('Confusion Matrix')
    plt.savefig(f"confusion_matrix_{learning_rate}_{num_epochs}.png", dpi=300)

def visualize_reconstructions(model, test_loader, learning_rate, num_epochs):
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.cuda()
            recon = model(data)
            break
    recon_error = ((data - recon)**2).mean(axis=1)
    vmax_global = torch.max(recon_error).item()
    plt.figure(figsize=(15, 15))
    _, ax = plt.subplots(3, 3)
    for i in range(3):
        ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)).clip(0, 1))
        ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)).clip(0, 1))
        ax[2, i].imshow(recon_error[i].cpu().numpy(), cmap='jet', vmax=vmax_global)
        ax[0, i].axis('off')
        ax[1, i].axis('off')
        ax[2, i].axis('off')
    plt.suptitle('Reconstructions: Original | Reconstructed | Error Map')
    plt.savefig(f"reconstructions_{learning_rate}_{num_epochs}.png", dpi=300)

def main(learning_rate=1e-3, num_epochs=50):
    print(f"Training with learning rate: {learning_rate}, num_epochs: {num_epochs}")
    transform = get_transform()
    BS = 16
    train_dataset = CustomDataset(root_dir='dataset', mode='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    test_dataset = CustomDataset(root_dir='dataset', mode='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

    model = Autoencoder().cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    loss_values = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)
    torch.save(model.state_dict(), f"autoencoder_model_{learning_rate}_{num_epochs}.pth")

    y_true, y_pred, y_score = evaluate_model(model, test_loader)

    auc_roc_score = roc_auc_score(y_true, y_score)
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"AUC-ROC Score: {auc_roc_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    plot_roc_curve(y_true, y_score, auc_roc_score, learning_rate, num_epochs)
    plot_confusion_matrix(y_true, y_pred, learning_rate, num_epochs)
    visualize_reconstructions(model, test_loader, learning_rate, num_epochs)
    
    return loss_values

if __name__ == "__main__":
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    num_epochs = 50
    all_loss_values = []
    for lr in learning_rates:
        loss_values = main(lr, num_epochs)
        all_loss_values.append((lr, loss_values))
    
    # Plotting the loss values
    plt.figure()
    for lr, loss_values in all_loss_values:
        plt.plot(range(1, num_epochs + 1), loss_values, label=f'LR={lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig('training_loss.png', dpi=300)