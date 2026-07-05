"""KVキャッシュを使用するSelf-Attention"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CachedSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cache=None):
        """
        x: (batch, new_tokens, embed_dim) - キャッシュ使用時、新しく追加されるトークンのみが入力される
        cache: {"k": (batch, num_heads, past_len, head_dim), "v": ...} または None
        """
        batch_size, new_len, embed_dim = x.shape

        qkv = self.qkv_proj(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        if cache is not None:
            # 過去のK, Vの後に、新たに計算したK, Vを結合する
            K = torch.cat([cache["k"], K], dim=2)
            V = torch.cat([cache["v"], V], dim=2)

        new_cache = {"k": K, "v": V}  # 次のステップで再利用するキャッシュ

        past_len = K.shape[2] - new_len
        total_len = K.shape[2]

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

        # 新しいトークンは「自分自身とその前にあるすべてのトークン」のみを参照できる必要がある
        mask = torch.ones(new_len, total_len, device=x.device)
        for i in range(new_len):
            mask[i, past_len + i + 1:] = 0
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(batch_size, new_len, embed_dim)
        return self.out_proj(out), new_cache
