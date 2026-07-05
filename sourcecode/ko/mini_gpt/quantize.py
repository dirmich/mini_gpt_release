"""가중치 양자화 기본 구현: INT8 대칭 양자화"""
import torch


def quantize_int8_symmetric(weight):
    """대칭 양자화: 0을 중심으로 -127~127 범위에 매핑"""
    max_abs = weight.abs().max()
    scale = max_abs / 127.0

    quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def dequantize_int8_symmetric(quantized, scale):
    return quantized.float() * scale


def quantization_error(original, dequantized):
    """양자화로 인한 오차를 정량적으로 측정"""
    mse = ((original - dequantized) ** 2).mean().item()
    max_error = (original - dequantized).abs().max().item()
    return mse, max_error
