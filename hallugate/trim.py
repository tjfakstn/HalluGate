"""근거 다이어트 (SPEC §2.9): 긴 근거에서 질문과 관련된 조각만 골라 LLM에 보내고 같은 조각으로 판정한다.

조각 나누기는 컬렉션(store.split_words)과 같고, 임베더도 컬렉션 검색과 같은 것을 쓴다.
고른 조각은 원문 순서대로 돌려준다(문서 흐름 유지).
"""
import numpy as np

from .store import split_words


class Trimmer:
    def __init__(self, embedder, chunk_words=120, overlap=20):
        self.embedder, self.chunk_words, self.overlap = embedder, chunk_words, overlap

    def _embed(self, texts):
        v = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(v, dtype=np.float32).reshape(len(texts), -1)

    def select(self, reference, query, k):
        """(고른 조각 리스트(원문 순서), 전체 조각 수). 조각이 k개 이하이거나 질문이 비면 전체를 그대로."""
        chunks = split_words(reference, self.chunk_words, self.overlap) or [reference]
        if len(chunks) <= k or not (query or '').strip():
            return chunks if len(chunks) > 1 else [reference], len(chunks)
        sims = self._embed(chunks) @ self._embed([query])[0]
        top = sorted(np.argsort(-sims)[:k].tolist())
        return [chunks[i] for i in top], len(chunks)
