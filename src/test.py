import torch
from pathlib import Path
from anomalib.data import PredictDataset
from anomalib.engine import Engine

torch.cuda.empty_cache()

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.pth', map_location=device, weights_only=False)
model.eval()
model.to(device)

# Initialize the engine (handles device automatically)
engine = Engine()

# Path to your test image
image_path = Path("dataset/Defaut/image_2_ST_Inf.png")

# Create dataset for single image
dataset = PredictDataset(
    path=image_path,
    image_size=(256, 256),  # Must match training size
)

# Get predictions
predictions = engine.predict(
    model=model,
    dataset=dataset,
)

# Process results
if predictions:
    # Access the prediction object
    prediction = predictions[0]  # Assuming predictions is a list with one element

    # Extract the predicted label
    pred_label = prediction["pred_labels"].item()  # Convert tensor to Python boolean
    print(f"Predicted Label: {'Anomaly' if pred_label else 'Normal'}")

    # Extract the confidence score
    pred_score = prediction["pred_scores"].item()  # Convert tensor to Python float
    print(f"Confidence Score: {pred_score:.4f}")