import os
import zipfile
import torch
from anomalib.data.image.folder import Folder
from anomalib.models.image import Patchcore
from anomalib.engine import Engine

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def initialize_datamodule(dataset_root):
    return Folder(
        name="dataset",
        root=dataset_root,
        normal_dir="Sans_Defaut",
        abnormal_dir="Defaut",
        normal_split_ratio=0.2,
        val_split_mode="from_test",
        val_split_ratio=0.5,
        image_size=(32, 32),
        train_batch_size=4,
        eval_batch_size=4,
        task="classification",
        num_workers=4
    )

def initialize_model():
    return Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.001
    )

def initialize_engine():
    return Engine(
        max_epochs=50,
        devices="cuda:0",
        pixel_metrics=["AUROC"],
        task="classification",
    )

def train_model(engine, datamodule, model):
    torch.cuda.empty_cache()
    engine.fit(datamodule=datamodule, model=model)
    torch.save(model, 'model2.pth')

def main(binary=True):
    dataset_root = "dataset" if binary else "dataset_multi"
    datamodule = initialize_datamodule(dataset_root)
    model = initialize_model()
    engine = initialize_engine()
    train_model(engine, datamodule, model)

if __name__ == "__main__":
    main()