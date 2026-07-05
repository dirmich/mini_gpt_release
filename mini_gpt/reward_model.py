"""보상모델: 언어모델 위에 스칼라 점수를 출력하는 헤드를 하나 얹은 것"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        # 마지막 은닉 상태를 스칼라 점수 하나로 변환하는 작은 헤드
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)

        # 각 시퀀스의 "실제 마지막 토큰" 위치의 은닉 상태를 사용 (패딩 제외)
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        last_token_hidden = last_hidden[batch_indices, seq_lengths]

        reward = self.value_head(last_token_hidden).squeeze(-1)  # (batch,)
        return reward


def reward_model_loss(chosen_rewards, rejected_rewards):
    """chosen 점수가 rejected 점수보다 높아지도록 유도하는 손실 (Bradley-Terry 모델)"""
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
