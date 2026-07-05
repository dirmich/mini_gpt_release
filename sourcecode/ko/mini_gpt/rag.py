"""RAG 파이프라인: 청킹, 임베딩, 검색, 리랭킹"""
import numpy as np


def chunk_text(text, chunk_size=300, overlap=50):
    """단어 수 기준으로 겹치게 텍스트를 나눈다."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # overlap만큼 겹치게 다음 청크 시작
    return chunks


class SimpleVectorStore:
    """FAISS 없이도 이해할 수 있는 최소 벡터 검색 구현 (코사인 유사도)"""

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
        # 코사인 유사도 = 정규화된 벡터끼리의 내적
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        similarities = (self.vectors @ query_vec) / (norms + 1e-8)

        top_indices = np.argsort(-similarities)[:top_k]
        return [(self.texts[i], float(similarities[i])) for i in top_indices]
