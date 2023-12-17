import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, embed_dim: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, embed_dim))

    def forward(self, tokens):
        """
        Args:
            tokens: (batch_size, seq_len)
        """
        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x


class CLIPLayer(nn.Module):
    def __init__(self, n_heads: int, embed_dim: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(n_heads, embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.linear_1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.linear_2 = nn.Linear(4 * embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, dim)
        """
        residue = x

        ## Self Attention
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)

        x += residue

        ## Feed Forward
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)

        x = F.gelu(x)
        x = self.linear_2(x)
        x += residue

        return x


class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Module([
            CLIPLayer(12, 768) for _ in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)        
        output = self.layernorm(state)

        return output