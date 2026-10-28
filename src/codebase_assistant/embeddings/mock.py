"""Deterministic mock embedding provider for fast offline development and testing."""

import hashlib
import math
import re
from typing import List

from codebase_assistant.embeddings.base import BaseEmbeddingProvider
from codebase_assistant.retrieval.tokenizer import CodeTokenizer


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, token-sensitive pseudo-embedding provider.

    Produces normalized float vectors using hash-based random projections.
    Texts sharing tokens produce higher cosine similarities than unrelated texts,
    enabling realistic retrieval testing without external API calls or network access.
    """

    def __init__(self, dimension: int = 768, model_name: str = "mock-embedding-v1") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        self._dimension = dimension
        self._model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def _hash_token(self, token: str) -> List[float]:
        """Generate a deterministic pseudo-random projection for a single token."""
        vec = [0.0] * self._dimension
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        seed_int = int.from_bytes(digest[:8], byteorder="big")

        # Linear Congruential Generator for reproducible pseudo-random values
        state = seed_int & 0xFFFFFFFFFFFFFFFF
        for i in range(self._dimension):
            state = (6364136223846793005 * state + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            # Map to range [-1.0, 1.0]
            val = ((state >> 32) & 0xFFFFFFFF) / 0xFFFFFFFF
            vec[i] = (val * 2.0) - 1.0
        return vec

    def _normalize(self, vec: List[float]) -> List[float]:
        """Normalize vector to unit length (L2 norm = 1.0)."""
        squared_sum = sum(x * x for x in vec)
        norm = math.sqrt(squared_sum)
        if norm == 0.0:
            # Fallback for all-zero vectors: return unit vector on first axis
            res = [0.0] * self._dimension
            res[0] = 1.0
            return res
        return [round(x / norm, 7) for x in vec]

    def embed_text(self, text: str) -> List[float]:
        """Generate a deterministic unit-length embedding vector for the given text."""
        cleaned = text.strip()
        if not cleaned:
            res = [0.0] * self._dimension
            res[0] = 1.0
            return res

        # Code-aware tokenization (splits camelCase, snake_case, etc.)
        tokens = CodeTokenizer.tokenize(cleaned)
        if not tokens:
            tokens = [cleaned.lower()]

        accum = [0.0] * self._dimension
        for token in tokens:
            token_proj = self._hash_token(token)
            for i in range(self._dimension):
                accum[i] += token_proj[i]

        return self._normalize(accum)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of text pieces."""
        return [self.embed_text(t) for t in texts]
