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
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.norm1 = norm(hidden_dim)
        self.attn = Attention(hidden_dim, head_dim, num_heads)
        self.norm2 = norm(hidden_dim)
        self.mlp = MLP(hidden_dim, mlp_dim, activation)

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


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        hidden_dim: int,
        head_dim: int,
        num_heads: int,
        mlp_dim: int,
        norm: nn.Module = nn.RMSNorm,
        activation: Callable = nn.functional.silu,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.encoder = nn.Embedding(vocab_size, hidden_dim)
        self.model = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    norm=norm,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = nn.Sequential(
            norm(hidden_dim), nn.Linear(hidden_dim, vocab_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        for block in self.model:
            x = block(x)
        x = self.decoder(x)
        return x
