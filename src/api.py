from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from prepare_dataset import extractAndOrder
from train import main
from pathlib import Path
from anomalib.data import PredictDataset
from anomalib.engine import Engine
import torch
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image

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


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    
    file = request.files['file']
    try:
        binary = bool(request.json.get('binary', True))
    except Exception as e:
        return jsonify({"status": "error", "message": "Error while getting binary status"}), 400

    # Select appropriate model and engine
    model = binary_model if binary else binary_model # TODO
    engine = binary_engine if binary else binary_engine # TODO
    
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
        
        result = {
            "label": "Anomaly" if prediction["pred_labels"].item() else "Normal",
            "confidence": round(prediction["pred_scores"].item(), 4),
            "heatmap": generate_heatmap(prediction["anomaly_maps"])
        }

        return jsonify({"status": "success", "result": result})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)