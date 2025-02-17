import torch
from pathlib import Path
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt
import numpy as np

torch.cuda.empty_cache()

# Defect type mapping
DEFECT_TYPES = {
    0: "Sans_Defaut",
    1: "SL",
    2: "ST_Inf", 
    3: "ST_Sup",
    4: "ST_Sup_Pli",
    5: "STP"
}

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('models/model_bin.pth', map_location=device, weights_only=False)
model.eval()
model.to(device)

# Initialize the engine (handles device automatically)
engine = Engine()

# Path to your test dataset
dataset_path = Path("data/dataset/")

# Create dataset for all images
dataset = PredictDataset(
    path=dataset_path,
    image_size=(256, 256),  # Must match training size
)

# Collect predictions and ground truth labels
all_preds = []
all_labels = []

for image_path in dataset_path.glob("**/*.png"):
    dataset = PredictDataset(
        path=image_path,
        image_size=(256, 256),  # Must match training size
    )
    predictions = engine.predict(
        model=model,
        dataset=dataset,
    )
    if predictions:
        prediction = predictions[0]  # Assuming predictions is a list with one element
        pred_class = prediction["pred_labels"].item()
        true_class = int(image_path.parent.name.split('_')[0])  # Assuming folder name contains the true class

        all_preds.append(pred_class)
        all_labels.append(true_class)

# Compute confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(conf_matrix)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(DEFECT_TYPES)
all_labels_one_hot = np.eye(n_classes)[all_labels]
all_preds_one_hot = np.eye(n_classes)[all_preds]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_one_hot[:, i], all_preds_one_hot[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {DEFECT_TYPES[i]} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Print classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=list(DEFECT_TYPES.values())))