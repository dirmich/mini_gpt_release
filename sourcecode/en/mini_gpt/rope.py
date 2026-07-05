"""RoPE (Rotary Position Embedding) Implementation"""
import torch


def build_rope_cache(head_dim, max_seq_len, base=10000.0, device="cpu"):
    """Precompute rotation angle tables (cos, sin) for each position."""
    # head_dim must be even (as we rotate pairs of elements)
    assert head_dim % 2 == 0

    # Use different rotation speeds (frequencies) per dimension: low dimensions rotate faster, high dimensions rotate slower
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()

    # angles[pos, i] = position * freqs[i]
    angles = torch.outer(positions, freqs)  # (max_seq_len, head_dim/2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin  # Each is (max_seq_len, head_dim/2)


def apply_rope(x, cos, sin):
    """Apply rotation to x: (batch, num_heads, seq_len, head_dim)."""
    seq_len = x.shape[-2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,head_dim/2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # Split x into two halves (x1, x2) to form pairs - the two axes for 2D plane rotation
    x1, x2 = x[..., 0::2], x[..., 1::2]

    # 2D Rotation Transformation: [x1', x2'] = [x1*cos - x2*sin, x1*sin + x2*cos]
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos

    # Reassemble into the original dimension order
    out = torch.stack([rotated_x1, rotated_x2], dim=-1)
    return out.flatten(-2)
