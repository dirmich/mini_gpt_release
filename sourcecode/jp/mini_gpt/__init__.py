"""mini_gpt: LLM基礎学習用ミニGPT実装パッケージ

このパッケージは、書籍「LLM基礎：ゼロから作る大規模言語モデル」の
第2〜8章で扱ったコードをそのままモジュール化したものです。

使用例:
    from mini_gpt.tokenizer import BPETokenizer
    from mini_gpt.model import MiniGPT
    from mini_gpt.generate import generate
"""

from mini_gpt.tokenizer import BPETokenizer
from mini_gpt.embedding import GPTEmbedding
from mini_gpt.attention import SelfAttention, MultiHeadAttention
from mini_gpt.transformer_block import FeedForward, TransformerBlock
from mini_gpt.model import MiniGPT
from mini_gpt.dataset import TextDataset
from mini_gpt.train import train_model
from mini_gpt.generate import generate

__all__ = [
    "BPETokenizer",
    "GPTEmbedding",
    "SelfAttention",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "MiniGPT",
    "TextDataset",
    "train_model",
    "generate",
]

__version__ = "1.0.0"
