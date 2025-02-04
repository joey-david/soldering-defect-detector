import torch
from anomalib.deploy import TorchInferencer
import cv2

def load_and_infer_model(model_path):
    torch.cuda.empty_cache()
    loaded_model = torch.load(model_path)
    inferencer_model = TorchInferencer(model=loaded_model)
    return inferencer_model

# Example usage
model_path = "model.pth"
inferencer_model = load_and_infer_model(model_path)