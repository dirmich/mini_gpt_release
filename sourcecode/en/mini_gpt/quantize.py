"""Basic Weight Quantization Implementation: INT8 Symmetric Quantization"""
import torch


def quantize_int8_symmetric(weight):
    """Symmetric Quantization: Mapping to the range -127 to 127 centered around 0"""
    max_abs = weight.abs().max()
    scale = max_abs / 127.0

    quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def dequantize_int8_symmetric(quantized, scale):
    return quantized.float() * scale


def quantization_error(original, dequantized):
    """Quantitatively measure errors caused by quantization"""
    mse = ((original - dequantized) ** 2).mean().item()
    max_error = (original - dequantized).abs().max().item()
    return mse, max_error
