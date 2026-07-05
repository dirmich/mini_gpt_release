"""Selective State Space Model의 최소 구현 (교육용, 순수 재귀 방식)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    def __init__(self, embed_dim, state_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim

        # 입력에 따라 B, C, 그리고 상태 유지 정도(delta)를 동적으로 계산하는 작은 선형층들
        self.delta_proj = nn.Linear(embed_dim, embed_dim)
        self.B_proj = nn.Linear(embed_dim, state_dim)
        self.C_proj = nn.Linear(embed_dim, state_dim)

        # A는 채널별로 학습되는 고정 파라미터 (음수로 유지해 상태가 발산하지 않게 함)
        self.A_log = nn.Parameter(torch.randn(embed_dim, state_dim) * 0.1)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape

        delta = F.softplus(self.delta_proj(x))       # (batch, seq_len, embed_dim), 항상 양수
        B = self.B_proj(x)                              # (batch, seq_len, state_dim)
        C = self.C_proj(x)                                # (batch, seq_len, state_dim)
        A = -torch.exp(self.A_log)                          # (embed_dim, state_dim), 항상 음수

        # 상태를 이산 시간으로 변환 (zero-order hold 이산화, 원 논문 방식을 간략화)
        # deltaA: (batch, seq_len, embed_dim, state_dim)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)

        h = torch.zeros(batch_size, embed_dim, self.state_dim, device=x.device)
        outputs = []
        for t in range(seq_len):
            # 재귀 업데이트: h_t = A_disc * h_{t-1} + B_disc * x_t
            h = deltaA[:, t] * h + deltaB_x[:, t]
            # 출력: y_t = C_t · h_t  (state_dim 축으로 합산)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (batch, embed_dim)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (batch, seq_len, embed_dim)


class MambaBlock(nn.Module):
    """SSM을 감싸는 블록: 트랜스포머 블록과 같은 자리에 끼워 넣을 수 있는 형태"""

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

        # 짧은 지역적 문맥을 섞어주는 causal 1D convolution
        x_conv = self.conv(x.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)
        x = F.silu(x_conv)

        x = self.ssm(x)
        x = self.out_proj(x)
        return residual + x
