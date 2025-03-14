import torch
import torch.nn as nn
from torch.nn.attention import flex_attention

from typing import Callable, Optional


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.activation = activation

        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.fc2 = nn.Linear(hidden_dim, mlp_dim)
        self.fc3 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.activation(self.fc1(x))
        x2 = self.fc2(x)
        x = x1 * x2
        x = self.fc3(x)
        return x
