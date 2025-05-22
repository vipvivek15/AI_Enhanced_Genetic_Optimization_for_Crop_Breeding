import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import CropYieldMLP
import argparse
import os
import joblib

def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.drop(columns=['plant_id', 'location_id'], errors='ignore')

    X = df.drop('trait_yield', axis=1).values
    y = df['trait_yield'].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y

def train_model(X, y, epochs, batch_size, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

    model = CropYieldMLP(input_dim=X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"[INFO] Training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch+1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor.to(device))
                val_loss = criterion(val_outputs, y_val_tensor.to(device))
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss.item():.4f}")

    print("[SUCCESS] Training complete.")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to cleaned_data.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_path", type=str, default="models/trained_model.pth")

    args = parser.parse_args()

    X, y, scaler_X, scaler_y = load_data(args.data)
    trained_model = train_model(X, y, args.epochs, args.batch_size, args.lr)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(trained_model.state_dict(), args.save_path)
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    print(f"[INFO] Model saved to {args.save_path}")
    print(f"[INFO] Scaler saved to models/scaler_X.pkl")
