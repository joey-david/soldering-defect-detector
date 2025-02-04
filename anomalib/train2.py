from anomalib.data.image.folder import Folder
from anomalib.models.image import Patchcore
from anomalib.engine import Engine
import torch
import zipfile
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


dataset_root = "dataset"

# Initialize datamodule with the correct root path
datamodule = Folder(
    name="dataset",
    root=dataset_root,  # Use the absolute path to the dataset
    normal_dir="Sans_Defaut",
    abnormal_dir="Defaut",
    normal_split_ratio=0.2,
    val_split_mode="from_test",
    val_split_ratio=0.5,
    image_size=(256, 256),
    train_batch_size=4, # default value
    eval_batch_size=4, # default value
    task="classification",
    num_workers=4
    )

# Model configuration
model = Patchcore(
    backbone="resnet34",
    layers=["layer2", "layer3"]
)

# Training setup
engine = Engine(
    max_epochs=50,
    devices=1,
    pixel_metrics=["AUROC", "F1Score"],
    task="classification"
)

# Start training
engine.fit(datamodule=datamodule, model=model)

# Save the model
torch.save(model.state_dict(), 'model.pth')
