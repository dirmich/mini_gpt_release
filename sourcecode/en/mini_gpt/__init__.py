"""mini_gpt: A mini GPT implementation package for learning LLM fundamentals

This package is a modularized version of the code covered in chapters
2 through 8 of the book "LLM Fundamentals: Building Large Language Models from Scratch."

Usage example:
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
