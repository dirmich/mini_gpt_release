"""トークン埋め込み + 位置埋め込み"""
import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, dropout=0.1):
        super().__init__()
        # トークンID -> ベクトル (ルックアップテーブル), サイズ: (vocab_size, embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # 位置インデックス -> ベクトル, サイズ: (max_seq_len, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len)
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        # positions: (1, seq_len) -> ブロードキャストによりバッチ全体に適用

        token_vecs = self.token_embedding(token_ids)      # (batch, seq_len, embed_dim)
        position_vecs = self.position_embedding(positions)  # (1, seq_len, embed_dim)

        # 2つのベクトルを加算することで「このトークンであり、かつこの位置である」という情報を共に保持する
        embeddings = token_vecs + position_vecs
        return self.dropout(embeddings)
