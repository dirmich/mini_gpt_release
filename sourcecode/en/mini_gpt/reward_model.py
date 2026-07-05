"""Reward Model: A head placed on top of a language model that outputs a scalar score"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        # A small head that converts the last hidden state into a single scalar score
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

        # Uses the hidden state at the "actual last token" position of each sequence (excluding padding)
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        last_token_hidden = last_hidden[batch_indices, seq_lengths]

        reward = self.value_head(last_token_hidden).squeeze(-1)  # (batch,)
        return reward


def reward_model_loss(chosen_rewards, rejected_rewards):
    """Loss that encourages chosen scores to be higher than rejected scores (Bradley-Terry model)"""
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
