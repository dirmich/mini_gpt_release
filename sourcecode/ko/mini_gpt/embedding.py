"""토큰 임베딩 + 위치 임베딩"""
import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, dropout=0.1):
        super().__init__()
        # 토큰 ID -> 벡터 (조회 테이블), 크기: (vocab_size, embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # 위치 인덱스 -> 벡터, 크기: (max_seq_len, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len)
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        # positions: (1, seq_len) -> 브로드캐스팅으로 배치 전체에 적용

        token_vecs = self.token_embedding(token_ids)      # (batch, seq_len, embed_dim)
        position_vecs = self.position_embedding(positions)  # (1, seq_len, embed_dim)

        # 두 벡터를 더해서 "이 토큰이면서 동시에 이 위치"라는 정보를 함께 담는다
        embeddings = token_vecs + position_vecs
        return self.dropout(embeddings)
