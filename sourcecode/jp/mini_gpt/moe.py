"""Mixture-of-Experts: ルーター + 複数の FeedForward エキスパート"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """既存の FeedForward と同一構造のエキスパート"""

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
        x_flat = x.view(-1, embed_dim)  # (batch*seq_len, embed_dim) - トークン単位で処理

        router_logits = self.router(x_flat)  # (num_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # 各トークンごとにスコアが最も高い top_k 個のエキスパートを選択
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # 選択された top_k 個の確率のみを再正規化 (合計が 1 になるように)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        # 負荷分散確認用: エキスパートごとに何個のトークンが割り当てられたかを記録
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)

        for expert_idx in range(self.num_experts):
            # このエキスパートが top_k 内に選択されたすべての (トークン, 順位) 位置を特定する
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
    """エキスパートごとの割り当て比率と平均ルーティング確率の積を最小化するように誘導する。

    これにより、特定のエキスパートにトークンが集中することを抑制する効果がある。
    (Switch Transformer の論文で提案された方式を簡略化)
    """
    num_tokens = router_probs.shape[0]

    # エキスパートごとの実際の選択比率 (top_k のいずれかに選ばれたトークンの割合)
    expert_mask = F.one_hot(top_k_indices, num_classes=num_experts).float()  # (tokens, top_k, num_experts)
    tokens_fraction = expert_mask.sum(dim=(0, 1)) / (num_tokens * top_k_indices.shape[1])

    # エキスパートごとの平均ルーティング確率
    probs_fraction = router_probs.mean(dim=0)

    # 両値の内積にエキスパート数を乗算 -> 完全に均等な時に最小値(1.0)を持つようスケーリング
    loss = num_experts * (tokens_fraction * probs_fraction).sum()
    return loss
