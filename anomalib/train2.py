import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Construction de notre dataset à partir du dossier dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        if self.mode == 'train':
            self.images = []
            good_dir = os.path.join(root_dir, 'Sans_Defaut')
            for img_name in os.listdir(good_dir):
                if img_name.endswith('.png'):
                    self.images.append(os.path.join(good_dir, img_name))
        elif self.mode == 'test':
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
        else:
            raise ValueError("Mode should be 'train' or 'test'")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.mode == 'train':
            return image
        else:
            return image, self.labels[idx]

# Data loaders
BS = 16
train_dataset = CustomDataset(root_dir='dataset', mode='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)

test_dataset = CustomDataset(root_dir='dataset', mode='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder().cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for img in train_loader:
        img = img.cuda()
        output = model(img)
        loss = 100_000*criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation on test set
y_true = []
y_pred = []
y_score = []
model.eval()

# Define the best threshold for classification
best_threshold = 0.5  # You can adjust this threshold based on your requirements
with torch.no_grad():
    for data, label in test_loader:
        data = data.cuda()
        recon = model(data)
        y_score_batch = ((data - recon)**2).mean(axis=(1))[:, 0:-10, 0:-10].mean(axis=(1, 2)).cpu().numpy()
        y_pred_batch = (y_score_batch >= best_threshold).astype(int)
        y_true.extend(label.numpy().tolist())
        y_pred.extend(y_pred_batch.tolist())
        y_score.extend(y_score_batch.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.array(y_score)

# Evaluation metrics
auc_roc_score = roc_auc_score(y_true, y_score)
print(f"AUC-ROC Score: {auc_roc_score}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Visualize reconstructions
with torch.no_grad():
    for data, _ in test_loader:
        data = data.cuda()
        recon = model(data)
        break
recon_error =  ((data-recon)**2).mean(axis=1)
plt.figure(dpi=250)
fig, ax = plt.subplots(3, 3, figsize=(5*4, 4*4))
for i in range(3):
    ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
    ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
    ax[2, i].imshow(recon_error[i][0:-10,0:-10].cpu().numpy(), cmap='jet',vmax= torch.max(recon_error[i]))
    ax[0, i].axis('OFF')
    ax[1, i].axis('OFF')
    ax[2, i].axis('OFF')
plt.show()