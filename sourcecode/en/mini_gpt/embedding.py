"""Token Embedding + Positional Embedding"""
import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, dropout=0.1):
        super().__init__()
        # Token ID -> Vector (Lookup Table), Size: (vocab_size, embed_dim)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Position Index -> Vector, Size: (max_seq_len, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len)
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        # positions: (1, seq_len) -> Applied to the entire batch via broadcasting

        token_vecs = self.token_embedding(token_ids)      # (batch, seq_len, embed_dim)
        position_vecs = self.position_embedding(positions)  # (1, seq_len, embed_dim)

        # Adding these two vectors encodes both "this token" and "this position" information together
        embeddings = token_vecs + position_vecs
        return self.dropout(embeddings)
