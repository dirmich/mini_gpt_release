"""Grouped-Query Attention (그룹 수 조절로 MHA, GQA, MQA를 모두 표현 가능)"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_groups, max_seq_len, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_heads % num_kv_groups == 0, "헤드 수는 KV 그룹 수로 나누어 떨어져야 함"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = embed_dim // num_heads
        self.group_size = num_heads // num_kv_groups  # 그룹 하나가 담당하는 쿼리 헤드 수

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        # K, V는 그룹 수만큼만 생성 (훨씬 적은 파라미터, 훨씬 작은 캐시)
        self.k_proj = nn.Linear(embed_dim, num_kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_groups * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # K, V를 group_size번씩 반복해서 Q의 헤드 수와 맞춘다
        # (batch, num_kv_groups, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(out)
