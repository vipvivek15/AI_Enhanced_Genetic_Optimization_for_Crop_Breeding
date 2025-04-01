# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CropYieldMLP(nn.Module):
    def __init__(self, input_dim):
        super(CropYieldMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(32, 1)  # Output: 1 continuous value (yield)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.output(x)  # No activation for regression output
        return x

