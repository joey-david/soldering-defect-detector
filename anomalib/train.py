from pathlib import Path
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
import torch

# Define paths
root = Path("dataset")
normal_dir = root / "Sans_Defaut"
abnormal_dir = root / "Defaut"

# Initialize the dataset
datamodule = Folder(
    root=root,
    normal_dir=normal_dir,
    abnormal_dir=abnormal_dir,
    normal_split_ratio=0.8,  # 80% training, 20% validation for normal images
    image_size=(256, 256),  # Adjust based on your needs
    train_batch_size=32,
    eval_batch_size=32,
    num_workers=4
)

# Initialize the model
model = Patchcore(
    input_size=(256, 256),
    backbone="wide_resnet50_2",
    layers=['layer2', 'layer3'],
)

# Configure the training engine
engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    default_root_dir="./results",  # Where to save results
    devices=1
)

# Train the model
engine.fit(
    datamodule=datamodule,
    model=model,
)

# Run inference on test set
engine.test(
    datamodule=datamodule,
    model=model,
)