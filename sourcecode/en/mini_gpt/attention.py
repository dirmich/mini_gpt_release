"""Implementation of Self-Attention and Multi-Head Attention"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """The most basic single-head Self-Attention"""

    def __init__(self, embed_dim, head_dim, max_seq_len, dropout=0.1):
        super().__init__()
        # Linear transformations to create Q, K, V. Input and output dimensions may differ (head_dim)
        self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask (lower triangular matrix) to prevent looking at future tokens
        # Since GPT's purpose is "next word prediction," referencing future information would be cheating.
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        Q = self.query_proj(x)   # (batch, seq_len, head_dim)
        K = self.key_proj(x)     # (batch, seq_len, head_dim)
        V = self.value_proj(x)   # (batch, seq_len, head_dim)

        head_dim = Q.shape[-1]
        # Similarity scores: (batch, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)

        # Masking future positions with -inf -> becomes 0 probability after softmax
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # Sum of each row is 1
        attn_weights = self.dropout(attn_weights)

        output = attn_weights @ V  # (batch, seq_len, head_dim)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multiple heads calculate attention in parallel from different perspectives"""

    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # Calculate Q, K, V all at once (processed with a single linear layer for efficiency)
        qkv = self.qkv_proj(x)  # (batch, seq_len, embed_dim*3)
        Q, K, V = qkv.chunk(3, dim=-1)

        # (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        def split_heads(t):
            t = t.view(batch_size, seq_len, self.num_heads, self.head_dim)
            return t.transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ V  # (batch, num_heads, seq_len, head_dim)

        # Concatenate back to (batch, seq_len, embed_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(out)
