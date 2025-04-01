# validation.py

import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import CropYieldMLP

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['plant_id', 'location_id'], errors='ignore')

    X = df.drop('trait_yield', axis=1).values
    y = df['trait_yield'].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return train_test_split(X_scaled, y_scaled, test_size=0.2), scaler_y

def evaluate_ai_model(model_path, X_test, y_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CropYieldMLP(input_dim=X_test.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_tensor).numpy()

    return predictions

def evaluate_baseline(X_test, y_test):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    return reg.predict(X_test)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to cleaned_data.csv")
    parser.add_argument("--ai_model", type=str, required=True, help="Path to trained_model.pth")

    args = parser.parse_args()
    (X_train, X_test, y_train, y_test), scaler_y = load_and_prepare_data(args.data)

    print("[INFO] Evaluating traditional Linear Regression...")
    baseline_preds = evaluate_baseline(X_test, y_test)
    print(f"Baseline RMSE: {mean_squared_error(y_test, baseline_preds, squared=False):.4f}")
    print(f"Baseline R2 Score: {r2_score(y_test, baseline_preds):.4f}")

    print("[INFO] Evaluating AI model...")
    ai_preds = evaluate_ai_model(args.ai_model, X_test, y_test)
    print(f"AI Model RMSE: {mean_squared_error(y_test, ai_preds, squared=False):.4f}")
    print(f"AI Model R2 Score: {r2_score(y_test, ai_preds):.4f}")

