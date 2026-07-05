"""RoPE (Rotary Position Embedding) の実装"""
import torch


def build_rope_cache(head_dim, max_seq_len, base=10000.0, device="cpu"):
    """位置ごとの回転角度 (cos, sin) テーブルを事前に計算しておく。"""
    # head_dim は偶数である必要がある (2つずつペアにして回転させるため)
    assert head_dim % 2 == 0

    # 次次元ごとに異なる回転速度 (周波数) を使用: 低次次元は速く、高次次元は遅く回転
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()

    # angles[pos, i] = position * freqs[i]
    angles = torch.outer(positions, freqs)  # (max_seq_len, head_dim/2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin  # それぞれ (max_seq_len, head_dim/2)


def apply_rope(x, cos, sin):
    """x: (batch, num_heads, seq_len, head_dim) に回転を適用する。"""
    seq_len = x.shape[-2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,head_dim/2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # x を半分ずつに分けて (x1, x2) のペアを作る - 2D 平面で回転させる2つの軸
    x1, x2 = x[..., 0::2], x[..., 1::2]

    # 2D 回転変換: [x1', x2'] = [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # 再び元の次元順序に並べ直す
    out = torch.stack([rotated_x1, rotated_x2], dim=-1)
    return out.flatten(-2)
