# app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, render_template
from deployment.server_config import FLASK_PORT, DEBUG_MODE

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models.model import CropYieldMLP

app = Flask(__name__)

# Define a dummy scaler and model (replace with actual fitted scaler and trained model later)
scaler = StandardScaler()
model = None
input_dim = 6  # Adjust based on number of features you're using

# Load trained model weights
def load_model():
    global model
    model = CropYieldMLP(input_dim=input_dim)
    model.load_state_dict(torch.load("models/trained_model.pth", map_location=torch.device('cpu')))
    model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Example: Assuming 6 input features sent from form or JSON
            input_data = [float(request.form.get(f"f{i}", 0)) for i in range(input_dim)]
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.fit_transform(input_array)  # NOTE: replace with actual fitted scaler

            with torch.no_grad():
                tensor = torch.tensor(input_scaled, dtype=torch.float32)
                prediction = model(tensor).item()

            return render_template("result.html", prediction=round(prediction, 2))
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict_api():
    try:
        data = request.json["features"]
        input_array = np.array(data).reshape(1, -1)
        input_scaled = scaler.fit_transform(input_array)  # Replace with saved scaler
        tensor = torch.tensor(input_scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(tensor).item()

        return jsonify({"predicted_yield": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    load_model()
    app.run(debug=DEBUG_MODE, port=FLASK_PORT)

