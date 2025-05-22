# app.py
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, render_template
from deployment.server_config import FLASK_PORT, DEBUG_MODE

import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from models.model import CropYieldMLP

app = Flask(__name__)

# Load pre-fitted scaler and trained model
scaler = joblib.load("models/scaler_X.pkl")
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
            input_data = [float(request.form.get(f"f{i+1}", 0)) for i in range(input_dim)]
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

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
        input_scaled = scaler.transform(input_array)
        tensor = torch.tensor(input_scaled, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(tensor).item()

        # Dummy confidence interval logic
        uncertainty_margin = 0.5
        lower_bound = prediction - uncertainty_margin
        upper_bound = prediction + uncertainty_margin

        return jsonify({
            "predicted_yield": prediction,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        })

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": str(e)}), 400

@app.route("/api/gpt", methods=["POST"])
def ask_gpt():
    try:
        question = request.json.get("question", "")
        if not question:
            return jsonify({"error": "No question provided."}), 400

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for crop science and agriculture."},
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )

        answer = response.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        print("GPT Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    app.run(debug=DEBUG_MODE, port=FLASK_PORT)