"""Transformerブロック: Attention + FeedForward + LayerNorm + Residual"""
import torch.nn as nn
from mini_gpt.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_mult=4, dropout=0.1):
        super().__init__()
        hidden_dim = embed_dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_mult=4, dropout=dropout)

    def forward(self, x):
        # Pre-LN構造: 正規化 -> サブリヤー -> 残差接続 (Residual Connection)
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x
