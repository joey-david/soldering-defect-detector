from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from prepare_dataset import extractAndOrder
from train import main
from pathlib import Path
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models.image.cfa.anomaly_map import AnomalyMapGenerator
import torch
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_file, send_from_directory
torch.cuda.empty_cache()

app = Flask(__name__, 
    static_folder='../static',
    template_folder='../templates'
)

CORS(app)

@app.route('/')
def home():
    return send_file('../templates/index.html')

# Initialize model and engine once at startup
def initialize_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    model.to(device)
    return model, Engine()

# Initialize both models at startup
binary_model, binary_engine = initialize_model('./models/model_bin.pth')
#multi_model, multi_engine = initialize_model('./models/model_bin.pth')

@app.route('/api/prepare-dataset', methods=['POST'])
def prepare_dataset():
    try:
        binary = bool(request.json.get('binary', True))
        extractAndOrder(binary)
        return jsonify({"status": "success", "message": "Dataset prepared successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def convert_heatmap_to_base64(anomaly_map):
    """Converts an anomaly heatmap into a base64 encoded image."""
    plt.figure(figsize=(4, 4))
    plt.imshow(anomaly_map, cmap='jet')
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    plt.close()

    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{base64_img}"


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    
    file = request.files['file']

    # Select appropriate model and engine
    model = binary_model
    engine = binary_engine
    
    try:
        temp_path = "data/temp.png"
        file.save(temp_path)
        
        # Create dataset and predict
        dataset = PredictDataset(
            path=Path(temp_path),
            image_size=(256, 256),
        )
        
        predictions = engine.predict(
            model=model,
            dataset=dataset,
        )

        os.remove(temp_path)
        
        if not predictions:
            return jsonify({"status": "error", "message": "No predictions generated"}), 500

        prediction = predictions[0]

            # Extract anomaly score map if available
        if "anomaly_maps" in prediction:
            anomaly_map = prediction["anomaly_maps"].squeeze().cpu().numpy()
        elif "distance" in prediction:  # Alternative key based on your previous error
            anomaly_map = prediction["distance"].squeeze().cpu().numpy()
        else:
            return jsonify({"status": "error", "message": "Anomaly map not found in prediction"}), 500

        # Convert the anomaly map to a base64 heatmap
        heatmap_img = convert_heatmap_to_base64(anomaly_map)

        result = {
            "label": "Anomaly" if prediction["pred_labels"].item() else "Normal",
            "confidence": round(prediction["pred_scores"].item(), 4),
            "heatmap": heatmap_img
        }

        return jsonify({"status": "success", "result": result})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)