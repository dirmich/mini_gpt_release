"""DPO 학습에 필요한 손실 함수를 직접 구현 (개념 이해용)"""
import torch
import torch.nn.functional as F


def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                      ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    각 인자는 (배치,) 크기의 텐서로, 해당 답변 시퀀스 전체의 로그확률 합을 의미한다.
    """
    policy_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps

    logits = beta * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits)

    # 모니터링용: 선택된 답변이 거부된 답변보다 얼마나 더 선호되고 있는지
    with torch.no_grad():
        chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
        accuracy = (chosen_reward > rejected_reward).float().mean()

    return loss.mean(), accuracy
