"""RAG Pipeline: Chunking, Embedding, Retrieval, Re-ranking"""
import numpy as np


def chunk_text(text, chunk_size=300, overlap=50):
    """Splits text into chunks based on word count with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # Start the next chunk with an overlap of 'overlap'
    return chunks


class SimpleVectorStore:
    """Minimal vector search implementation (Cosine Similarity) without FAISS"""

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
        # Cosine Similarity = Dot product of normalized vectors
        norms = np.linalg.norm(self.vectors, axis=1) * np.linalg.norm(query_vec)
        similarities = (self.vectors @ query_vec) / (norms + 1e-8)

        top_indices = np.argsort(-similarities)[:top_k]
        return [(self.texts[i], float(similarities[i])) for i in top_indices]
