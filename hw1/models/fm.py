import torch
from torch import nn


class FactorizationMachine(nn.Module):
    def __init__(self, n_features, n_factors=16, global_mean=0.0):
        super().__init__()

        self.bias = nn.Parameter(torch.tensor(float(global_mean)))
        self.linear = nn.Embedding(n_features, 1)
        self.factors = nn.Embedding(n_features, n_factors)

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, std=0.01)

    def forward(self, x):
        linear = self.linear(x).sum(dim=1).squeeze(-1)

        v = self.factors(x)

        interactions = 0.5 * (
            v.sum(dim=1).pow(2)
            - v.pow(2).sum(dim=1)
        ).sum(dim=1)

        return self.bias + linear + interactions
