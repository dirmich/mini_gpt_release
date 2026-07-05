"""モデル結合手法：重みの平均、Task Arithmetic、TIES"""
import copy
import torch


def average_merge(state_dicts):
    """複数の state_dict を単純平均する (Model Soup)"""
    merged = copy.deepcopy(state_dicts[0])
    for key in merged:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        merged[key] = stacked.mean(dim=0)
    return merged
def compute_task_vector(base_state_dict, finetuned_state_dict):
    """ファインチューニングによる重みの変化量（ベクトル）を計算"""
    return {
        key: finetuned_state_dict[key].float() - base_state_dict[key].float()
        for key in base_state_dict
    }


def apply_task_vectors(base_state_dict, task_vectors, scale=1.0):
    """複数のタスクベクトルを元のモデルに加算する（能力の合成）"""
    merged = copy.deepcopy(base_state_dict)
    for key in merged:
        combined_delta = sum(tv[key] for tv in task_vectors)
        merged[key] = merged[key].float() + scale * combined_delta
    return merged
def ties_merge(task_vectors, trim_ratio=0.2):
    """TIES-Merging: 符号の衝突を解決しながら複数のタスクベクトルを結合"""
    merged = {}
    keys = task_vectors[0].keys()

    for key in keys:
        vectors = torch.stack([tv[key] for tv in task_vectors])  # (num_models, *shape)

        # 1. Trim: 絶対値が小さい下位 trim_ratio 分の割合を 0 にする
        flat = vectors.abs().flatten()
        threshold = torch.quantile(flat, trim_ratio)
        trimmed = torch.where(vectors.abs() >= threshold, vectors, torch.zeros_like(vectors))

        # 2. Elect Sign: 符号ごとに絶対値の合計が大きい方を代表符号として選択
        positive_mass = torch.where(trimmed > 0, trimmed, torch.zeros_like(trimmed)).sum(dim=0)
        negative_mass = torch.where(trimmed < 0, trimmed, torch.zeros_like(trimmed)).sum(dim=0)
        elected_sign = torch.where(positive_mass.abs() >= negative_mass.abs(), 1.0, -1.0)

        # 3. Disjoint Merge: 代表符号と一致する値のみを集めて平均をとる
        agree_mask = (torch.sign(trimmed) == elected_sign.unsqueeze(0)) & (trimmed != 0)
        count = agree_mask.sum(dim=0).clamp(min=1)
        merged[key] = (trimmed * agree_mask).sum(dim=0) / count

    return merged
