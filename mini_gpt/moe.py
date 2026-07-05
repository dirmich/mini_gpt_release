"""Mixture-of-Experts: 라우터 + 여러 FeedForward 전문가"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """기존 FeedForward와 동일한 구조의 전문가 하나"""

    def __init__(self, embed_dim, hidden_mult=4, dropout=0.1):
        super().__init__()
        hidden_dim = embed_dim * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    def __init__(self, embed_dim, num_experts=8, top_k=2, hidden_mult=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(embed_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            Expert(embed_dim, hidden_mult, dropout) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.view(-1, embed_dim)  # (batch*seq_len, embed_dim) - 토큰 단위로 처리

        router_logits = self.router(x_flat)  # (num_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # 각 토큰마다 점수가 가장 높은 top_k개의 전문가를 선택
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # 선택된 top_k개 확률만 다시 정규화 (합이 1이 되도록)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        # 부하 분산 확인용: 전문가마다 몇 개의 토큰이 배정되었는지 기록
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)

        for expert_idx in range(self.num_experts):
            # 이 전문가가 top_k 안에 선택된 모든 (토큰, 순위) 위치를 찾는다
            mask = (top_k_indices == expert_idx)  # (num_tokens, top_k)
            if not mask.any():
                continue

            token_positions, k_positions = mask.nonzero(as_tuple=True)
            tokens_per_expert[expert_idx] = len(token_positions)

            expert_input = x_flat[token_positions]
            expert_output = self.experts[expert_idx](expert_input)

            weight = top_k_probs[token_positions, k_positions].unsqueeze(-1)
            output.index_add_(0, token_positions, expert_output * weight)

        output = output.view(batch_size, seq_len, embed_dim)
        return output, tokens_per_expert
def load_balancing_loss(router_probs, top_k_indices, num_experts):
    """전문가별 배정 비율과 평균 라우팅 확률의 곱을 최소화하도록 유도한다.

    이렇게 하면 특정 전문가에 토큰이 쏠리는 것을 억제하는 효과가 있다.
    (Switch Transformer 논문에서 제안된 방식을 간략화)
    """
    num_tokens = router_probs.shape[0]

    # 전문가별로 실제 선택된 비율 (top_k 중 하나로라도 뽑힌 토큰의 비율)
    expert_mask = F.one_hot(top_k_indices, num_classes=num_experts).float()  # (tokens, top_k, num_experts)
    tokens_fraction = expert_mask.sum(dim=(0, 1)) / (num_tokens * top_k_indices.shape[1])

    # 전문가별 평균 라우팅 확률
    probs_fraction = router_probs.mean(dim=0)

    # 두 값의 내적에 전문가 수를 곱함 -> 완전히 균등할 때 최소값(1.0)을 갖도록 스케일링
    loss = num_experts * (tokens_fraction * probs_fraction).sum()
    return loss
