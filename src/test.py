import torch
from pathlib import Path
from anomalib.data import PredictDataset
from anomalib.engine import Engine

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
model = torch.load('models/model_multi.pth', map_location=device, weights_only=False)
model.eval()
model.to(device)

# Initialize the engine (handles device automatically)
engine = Engine()

# Path to your test image
image_path = Path("data/dataset_multi/Defaut/ST_Sup_Pli/image_300_ST_Sup_Pli.png")

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
    pred_class = prediction["pred_labels"].item()

    pred_type = DEFECT_TYPES[pred_class]

    class_scores = prediction["pred_scores"]
    
    print(f"Predicted Type: {pred_type}")
    print("Class Confidence Scores:")
    for idx, score in enumerate(class_scores):
        print(f"{DEFECT_TYPES[idx]}: {score:.4f}")