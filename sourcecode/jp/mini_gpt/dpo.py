"""DPO学習に必要な損失関数を直接実装 (概念理解用)"""
import torch
import torch.nn.functional as F


def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                      ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    各引数は(batch,)サイズのテンソルであり、該当する回答シーケンス全体の対数確率の合計を意味する。
    """
    policy_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps

    logits = beta * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits)

    # モニタリング用: 選択された回答が拒否された回答よりもどれだけ好まれているか
    with torch.no_grad():
        chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
        accuracy = (chosen_reward > rejected_reward).float().mean()

    return loss.mean(), accuracy
