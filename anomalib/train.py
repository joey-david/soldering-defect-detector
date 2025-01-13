from anomalib.core.model import get_model
from anomalib.data import get_datamodule
from pytorch_lightning import Trainer
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("config.yaml")

# Initialize datamodule and model
datamodule = get_datamodule(config)
model = get_model(config)

# Train the model using PyTorch Lightning's Trainer
trainer = Trainer(**config.trainer)
trainer.fit(model=model, datamodule=datamodule)

