import torch
from torch import nn

class ParallelDecisionNet(nn.Module):
    """One forward pass predicts a joint distribution over four binary findings."""
    def __init__(self, input_dim, hidden=48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden),
            nn.SiLU(), nn.Dropout(0.15), nn.Linear(hidden, hidden // 2),
            nn.SiLU(), nn.Dropout(0.1), nn.Linear(hidden // 2, 16))

    def forward(self, x):
        return self.net(x)
