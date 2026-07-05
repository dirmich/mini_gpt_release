"""Self-Attention과 Multi-Head Attention 구현"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """가장 기본적인 단일 헤드 Self-Attention"""

    def __init__(self, embed_dim, head_dim, max_seq_len, dropout=0.1):
        super().__init__()
        # Q, K, V를 만드는 선형 변환. 입력과 출력 차원이 다를 수 있음(head_dim)
        self.query_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.key_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.value_proj = nn.Linear(embed_dim, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # 미래 토큰을 못 보게 막는 causal mask (하삼각행렬)
        # GPT는 "다음 단어 예측"이 목적이므로 미래 정보를 참조하면 반칙(치팅)이 됩니다.
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        Q = self.query_proj(x)   # (batch, seq_len, head_dim)
        K = self.key_proj(x)     # (batch, seq_len, head_dim)
        V = self.value_proj(x)   # (batch, seq_len, head_dim)

        head_dim = Q.shape[-1]
        # 유사도 점수: (batch, seq_len, seq_len)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)

        # 미래 위치는 -inf로 마스킹 -> softmax 후 확률이 0이 됨
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # 각 행의 합이 1
        attn_weights = self.dropout(attn_weights)

        output = attn_weights @ V  # (batch, seq_len, head_dim)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """여러 개의 헤드가 서로 다른 관점에서 병렬로 어텐션을 계산"""

    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim은 num_heads로 나누어 떨어져야 합니다"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # 한번에 Q, K, V를 모두 계산 (효율을 위해 하나의 선형층으로 처리)
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

        # 다시 (batch, seq_len, embed_dim)로 합치기
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.out_proj(out)
