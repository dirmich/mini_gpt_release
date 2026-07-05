"""mini_gpt: LLM 기초 학습용 미니 GPT 구현 패키지

이 패키지는 책 "LLM 기초: 밑바닥부터 만드는 대규모 언어모델"의
2~8장에서 다룬 코드를 그대로 모듈화한 것입니다.

사용 예:
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
