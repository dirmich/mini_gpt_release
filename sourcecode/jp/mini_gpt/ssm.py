"""Selective State Space Model の最小実装 (教育用、純粋な再帰方式)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    def __init__(self, embed_dim, state_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim

        # 入力に応じて B, C, および状態の保持度 (delta) を動的に計算する小さな線形層
        self.delta_proj = nn.Linear(embed_dim, embed_dim)
        self.B_proj = nn.Linear(embed_dim, state_dim)
        self.C_proj = nn.Linear(embed_dim, state_dim)

        # A はチャンネルごとに学習される固定パラメータ (負の値に保ち、状態の発散を防ぐ)
        self.A_log = nn.Parameter(torch.randn(embed_dim, state_dim) * 0.1)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape

        delta = F.softplus(self.delta_proj(x))       # (batch, seq_len, embed_dim), 常に正
        B = self.B_proj(x)                              # (batch, seq_len, state_dim)
        C = self.C_proj(x)                                # (batch, seq_len, state_dim)
        A = -torch.exp(self.A_log)                          # (embed_dim, state_dim), 常に負

        # 状態を離散時間へと変換 (zero-order hold 離散化、原論文の手法を簡略化)
        # deltaA: (batch, seq_len, embed_dim, state_dim)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)

        h = torch.zeros(batch_size, embed_dim, self.state_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            # 再帰更新: h_t = A_disc * h_{t-1} + B_disc * x_t
            h = deltaA[:, t] * h + deltaB_x[:, t]
            # 出力: y_t = C_t · h_t  (state_dim 軸で集約)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (batch, embed_dim)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, embed_dim)


class MambaBlock(nn.Module):
    """SSM をラップするブロック: Transformer ブロックと同じ位置に挿入可能な形式"""

    def __init__(self, embed_dim, state_dim=16, expand=2):
        super().__init__()
        inner_dim = embed_dim * expand
        self.in_proj = nn.Linear(embed_dim, inner_dim)
        self.conv = nn.Conv1d(inner_dim, inner_dim, kernel_size=3, padding=2, groups=inner_dim)
        self.ssm = SelectiveSSM(inner_dim, state_dim)
        self.out_proj = nn.Linear(inner_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.in_proj(x)  # (batch, seq_len, inner_dim)

        # 短い局所的なコンテキストを混合する causal 1D convolution
        x_conv = self.conv(x.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        x = F.silu(x_conv)

        x = self.ssm(x)
        x = self.out_proj(x)
        return residual + x
