import torch
import torch.nn as nn
from torch.nn.attention import flex_attention

from typing import Callable, Optional


class Attention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        head_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_heads = num_heads

        self.query = nn.Linear(hidden_dim, head_dim * num_heads)
        self.key = nn.Linear(hidden_dim, head_dim * num_heads)
        self.value = nn.Linear(hidden_dim, head_dim * num_heads)
        self.out = nn.Linear(head_dim * num_heads, hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[flex_attention.BlockMask] = None,
        cache=None,
    ):
        batch_size = query.size(0)

        key = key if key is not None else query
        value = value if value is not None else query

        query = (
            self.query(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.key(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.value(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        out = flex_attention.flex_attention(
            query=query,
            key=key,
            value=value,
            block_mask=mask,
        )

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.out(out)
        return out
