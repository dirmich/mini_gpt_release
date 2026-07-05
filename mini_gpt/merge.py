"""모델 병합 기법들: 가중치 평균, task arithmetic, TIES"""
import copy
import torch


def average_merge(state_dicts):
    """여러 state_dict를 단순 평균한다 (Model Soup)"""
    merged = copy.deepcopy(state_dicts[0])
    for key in merged:
        stacked = torch.stack([sd[key].float() for sd in state_dicts])
        merged[key] = stacked.mean(dim=0)
    return merged
def compute_task_vector(base_state_dict, finetuned_state_dict):
    """파인튜닝으로 인한 가중치 변화량(벡터)을 계산"""
    return {
        key: finetuned_state_dict[key].float() - base_state_dict[key].float()
        for key in base_state_dict
    }


def apply_task_vectors(base_state_dict, task_vectors, scale=1.0):
    """여러 task vector를 원본 모델에 더한다 (능력을 합성)"""
    merged = copy.deepcopy(base_state_dict)
    for key in merged:
        combined_delta = sum(tv[key] for tv in task_vectors)
        merged[key] = merged[key].float() + scale * combined_delta
    return merged
def ties_merge(task_vectors, trim_ratio=0.2):
    """TIES-Merging: 부호 충돌을 해결하며 여러 task vector를 병합"""
    merged = {}
    keys = task_vectors[0].keys()

    for key in keys:
        vectors = torch.stack([tv[key] for tv in task_vectors])  # (num_models, *shape)

        # 1. Trim: 절댓값이 작은 하위 trim_ratio 비율은 0으로
        flat = vectors.abs().flatten()
        threshold = torch.quantile(flat, trim_ratio)
        trimmed = torch.where(vectors.abs() >= threshold, vectors, torch.zeros_like(vectors))

        # 2. Elect Sign: 부호별로 절댓값 합이 더 큰 쪽을 대표 부호로 선택
        positive_mass = torch.where(trimmed > 0, trimmed, torch.zeros_like(trimmed)).sum(dim=0)
        negative_mass = torch.where(trimmed < 0, trimmed, torch.zeros_like(trimmed)).sum(dim=0)
        elected_sign = torch.where(positive_mass.abs() >= negative_mass.abs(), 1.0, -1.0)

        # 3. Disjoint Merge: 대표 부호와 일치하는 값들만 모아 평균
        agree_mask = (torch.sign(trimmed) == elected_sign.unsqueeze(0)) & (trimmed != 0)
        count = agree_mask.sum(dim=0).clamp(min=1)
        merged[key] = (trimmed * agree_mask).sum(dim=0) / count

    return merged
