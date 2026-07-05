"""ミニGPT: 埋め込み + トランスフォーマー・ブロック N個 + 出力層"""
import torch
import torch.nn as nn
from mini_gpt.embedding import GPTEmbedding
from mini_gpt.transformer_block import TransformerBlock


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4,
                 num_layers=4, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.embedding = GPTEmbedding(vocab_size, embed_dim, max_seq_len, dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        self.final_ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # 重みの共有: トークン埋め込みと出力層が同じ行列を使用
        self.lm_head.weight = self.embedding.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        # GPT-2に類似した初期化: 小さな標準偏差の正規分布
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):
        # token_ids: (batch, seq_len)
        x = self.embedding(token_ids)          # (batch, seq_len, embed_dim)

        for block in self.blocks:
            x = block(x)

        x = self.final_ln(x)
        logits = self.lm_head(x)               # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            # logits: (batch, seq_len, vocab_size) -> (batch*seq_len, vocab_size)
            # targets: (batch, seq_len) -> (batch*seq_len,)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
