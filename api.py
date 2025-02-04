from flask import Flask, request, jsonify, send_file
from prepare_dataset import extractAndOrder
from train import main
from test import predict_image
import os

app = Flask(__name__)

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/api/prepare-dataset', methods=['POST'])
def prepare_dataset():
    try:
        binary = request.json.get('binary', True)
        extractAndOrder(binary)
        return jsonify({"status": "success", "message": "Dataset prepared successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    try:
        main()
        return jsonify({"status": "success", "message": "Model trained successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400

    try:
        # Save uploaded file temporarily
        temp_path = "temp.png"
        file.save(temp_path)
        
        # Run prediction
        result = predict_image(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return jsonify({
            "status": "success",
            "prediction": result
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)