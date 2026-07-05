"""Direct implementation of the loss function required for DPO training (for conceptual understanding)"""
import torch
import torch.nn.functional as F


def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                      ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Each argument is a tensor of size (batch,), representing the sum of log probabilities for the entire response sequence.
    """
    policy_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps

    logits = beta * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits)


    with torch.no_grad():
        chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
        accuracy = (chosen_reward > rejected_reward).float().mean()

    return loss.mean(), accuracy
