"""Mixture-of-Experts: Router + Multiple FeedForward Experts"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A single expert with the same structure as a standard FeedForward layer"""

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
        x_flat = x.view(-1, embed_dim)  # (batch*seq_len, embed_dim) - Processed on a per-token basis

        router_logits = self.router(x_flat)  # (num_tokens, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # For each token, select the top_k experts with the highest scores
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # Re-normalize only the selected top_k probabilities (so they sum to 1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        # For load balancing: record how many tokens are assigned to each expert
        tokens_per_expert = torch.zeros(self.num_experts, device=x.device)

        for expert_idx in range(self.num_experts):
            # Find all (token, rank) positions where this expert was chosen within top_k
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
    """Induces minimization of the product between per-expert assignment ratio and average routing probability.

    This has an effect of suppressing tokens from clustering into a specific expert.
    (Simplified version of the method proposed in the Switch Transformer paper)
    """
    num_tokens = router_probs.shape[0]

    # Actual selection ratio per expert (ratio of tokens picked at least once among top_k)
    expert_mask = F.one_hot(top_k_indices, num_classes=num_experts).float()  # (tokens, top_k, num_experts)
    tokens_fraction = expert_mask.sum(dim=(0, 1)) / (num_tokens * top_k_indices.shape[1])

    # Average routing probability per expert
    probs_fraction = router_probs.mean(dim=0)

    # Multiply the dot product of these two values by the number of experts -> Scaled to have a minimum value (1.0) when perfectly uniform
    loss = num_experts * (tokens_fraction * probs_fraction).sum()
    return loss
