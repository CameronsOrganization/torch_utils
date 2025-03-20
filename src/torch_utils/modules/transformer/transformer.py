import torch
import torch.nn as nn
from torch.nn.attention import flex_attention

from typing import Callable, Optional

from .attention import Attention
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        head_dim: int,
        num_heads: int,
        mlp_dim: int,
        norm: nn.Module = nn.RMSNorm,
        mlp_activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.norm1 = norm(hidden_dim)
        self.attn = Attention(hidden_dim, head_dim, num_heads)
        self.norm2 = norm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_dim, mlp_activation)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        r = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + r
        r = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + r
        return x
