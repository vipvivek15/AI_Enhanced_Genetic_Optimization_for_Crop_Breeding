# hyperparameter_tuning.py

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from model import CropYieldMLP
import itertools
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['plant_id', 'location_id'], errors='ignore')
    
    X = df.drop('trait_yield', axis=1).values
    y = df['trait_yield'].values.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return train_test_split(X_scaled, y_scaled, test_size=0.2), X_scaled.shape[1]

class TunableMLP(nn.Module):
    def __init__(self, input_dim, layer_sizes):
        super(TunableMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.ReLU())
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_and_evaluate(X_train, X_val, y_train, y_val, input_dim, layer_sizes, lr, batch_size, epochs=20):
    model = TunableMLP(input_dim, layer_sizes)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.float32)),
                              batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
        return mean_squared_error(y_val, preds, squared=False)  # RMSE

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to cleaned_data.csv")
    args = parser.parse_args()

    (X_train, X_val, y_train, y_val), input_dim = load_data(args.data)

    learning_rates = [0.01, 0.001]
    batch_sizes = [16, 32]
    layer_configs = [[64, 32], [128, 64]]

    best_score = float('inf')
    best_config = None

    print("[INFO] Starting hyperparameter tuning...\n")
    for lr, bs, layers in itertools.product(learning_rates, batch_sizes, layer_configs):
        rmse = train_and_evaluate(X_train, X_val, y_train, y_val, input_dim, layers, lr, bs)
        print(f"Config: LR={lr}, Batch={bs}, Layers={layers} => RMSE={rmse:.4f}")
        if rmse < best_score:
            best_score = rmse
            best_config = (lr, bs, layers)

    print("\n✅ Best Configuration:")
    print(f"Learning Rate: {best_config[0]}")
    print(f"Batch Size: {best_config[1]}")
    print(f"Hidden Layers: {best_config[2]}")
    print(f"Lowest RMSE: {best_score:.4f}")

