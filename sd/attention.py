import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, dim: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(dim, 3 * dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(dim, dim, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = dim // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, dim)
        """

        input_shape = x.shape

        batch_size, seq_len, dim = input_shape

        intermidiate_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, 3 * dim) -> 3 tensors of shape (batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, n_heads, dim/n_heads) -> (batch_size, n_heads, seq_len, dim/n_heads)
        q = q.view(intermidiate_shape).transpose(1, 2)
        k = k.view(intermidiate_shape).transpose(1, 2)
        v = v.view(intermidiate_shape).transpose(1, 2)

        # (batch_size, n_heads, seq_len, seq_len)
        weights = torch.matmul(q, k.transpose(-1, -2))

        if causal_mask:
            mask = torch.ones_like(weights, dtype=torch.bool).triu(1)
            weights.masked_fill_(mask, -torch.inf)
        
        weights /= math.sqrt(self.d_head)

        weights = F.softmax(weights, dim=-1)

        # -> (batch_size, n_heads, seq_len, dim/n_heads)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2)
        output = output.reshape(input_shape)
        output = self.out_proj(output)

        # (batch_size, seq_len, dim)
        return output



class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, embed_dim: int, embed_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=in_proj_bias)
        self.k_proj = nn.Linear(embed_cross, embed_dim, bias=in_proj_bias)
        self.v_proj = nn.Linear(embed_cross, embed_dim, bias=in_proj_bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = embed_dim // n_heads

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            y: (batch_size, seq_len, embed_cross)
        """
        input_shape = x.shape
        batch_size, seq_len, embed_dim = input_shape

        intermed_shape = (batch_size, -1, self.n_heads, self.d_head)

        q = self.q_proj(x).view(intermed_shape).transpose(1, 2)
        k = self.k_proj(y).view(intermed_shape).transpose(1, 2)
        v = self.v_proj(y).view(intermed_shape).transpose(1, 2)

        weights = torch.matmul(q, k.transpose(-1, -2))
        weights /= math.sqrt(self.d_head)
        weights = F.softmax(weights, dim=-1)

        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous()
        output = output.reshape(input_shape)
        output = self.out_proj(output)

        return output