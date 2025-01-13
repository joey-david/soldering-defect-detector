# In anomalib/models/components/dimensionality_reduction/random_projection.py
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
import torch
from torch.utils.checkpoint import checkpoint
import os
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Define paths
root = Path("dataset")
normal_dir = "Sans_Defaut"
abnormal_dir = "Defaut"

# Initialize the dataset
datamodule = Folder(
    name="OUAIS",
    root=root,
    normal_dir=normal_dir,
    abnormal_dir=abnormal_dir,
    normal_split_ratio=0.8,
    image_size=(64, 64),
    train_batch_size=4,
    eval_batch_size=4,
    num_workers=4,
    test_split_mode="none"
)

# Modify the Patchcore model to use checkpointing and offload coreset index selection to CPU
class CheckpointedPatchcore(Patchcore):
    def forward(self, x):
        return checkpoint(super().forward, x)
    
    def select_coreset_indices(self, embeddings):
        # Move embeddings to CPU for coreset selection
        embeddings_cpu = embeddings.cpu()
        return super().select_coreset_indices(embeddings_cpu)

# Initialize the model
model = CheckpointedPatchcore(
    backbone="resnet18",
    layers=['layer2', 'layer3'],
    coreset_sampling_ratio=0.05,
)

model = model.to(memory_format=torch.channels_last, dtype=torch.float32)

# Configure the training engine with mixed-precision and gradient accumulation
engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    default_root_dir="./results",
    devices=1,
    accumulate_grad_batches=2  # Gradient accumulation over 2 batches
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