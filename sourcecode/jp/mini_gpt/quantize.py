"""重み量子化の基本実装: INT8 対称量子化"""
import torch


def quantize_int8_symmetric(weight):
    """対称量子化: 0を中心に -127〜127 の範囲にマッピング"""
    max_abs = weight.abs().max()
    scale = max_abs / 127.0

    quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return quantized, scale


def dequantize_int8_symmetric(quantized, scale):
    return quantized.float() * scale


def quantization_error(original, dequantized):
    """量子化による誤差を定量的に測定"""
    mse = ((original - dequantized) ** 2).mean().item()
    max_error = (original - dequantized).abs().max().item()
    return mse, max_error
