"""RoPE(Rotary Position Embedding) 구현"""
import torch


def build_rope_cache(head_dim, max_seq_len, base=10000.0, device="cpu"):
    """위치별 회전 각도(cos, sin) 테이블을 미리 계산해둔다."""
    # head_dim은 짝수여야 함 (2개씩 짝지어 회전시키기 때문)
    assert head_dim % 2 == 0

    # 차원마다 다른 회전 속도(주파수)를 사용: 낮은 차원은 빠르게, 높은 차원은 느리게 회전
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()

    # angles[pos, i] = position * freqs[i]
    angles = torch.outer(positions, freqs)  # (max_seq_len, head_dim/2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin  # 각각 (max_seq_len, head_dim/2)


def apply_rope(x, cos, sin):
    """x: (batch, num_heads, seq_len, head_dim)에 회전을 적용한다."""
    seq_len = x.shape[-2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,head_dim/2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # x를 절반씩 나눠 (x1, x2) 짝을 만든다 - 2차원 평면에서 회전시킬 두 축
    x1, x2 = x[..., 0::2], x[..., 1::2]

    # 2D 회전 변환: [x1', x2'] = [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # 다시 원래 차원 순서로 끼워 맞추기
    out = torch.stack([rotated_x1, rotated_x2], dim=-1)
    return out.flatten(-2)
