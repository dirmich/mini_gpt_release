"""RAGパイプライン：チャンキング、埋め込み、検索、リランキング"""
import numpy as np


def chunk_text(text, chunk_size=300, overlap=50):
    """単語数に基づいてテキストを重複させながら分割する。"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap分だけ重複させて次のチャンクを開始
    return chunks


class SimpleVectorStore:
    """FAISSなしでも理解できる最小限のベクトル検索の実装 (コサイン類似度)"""

    def __init__(self, embed_fn):
        self.embed_fn = embed_fn
        self.texts = []
        self.vectors = None

    def add(self, texts):
        self.texts.extend(texts)
        new_vectors = self.embed_fn(texts)
        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

    def search(self, query, top_k=3):
        query_vec = self.embed_fn([query])[0]
        # コサイン類似度 = 正規化されたベクトル同士の内積
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        similarities = (self.vectors @ query_vec) / (norms + 1e-8)

        top_indices = np.argsort(-similarities)[:top_k]
        return [(self.texts[i], float(similarities[i])) for i in top_indices]
