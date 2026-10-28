"""Gemini embedding provider using official google-genai SDK."""

import logging
import math
from typing import List, Optional

from codebase_assistant.config import get_settings
from codebase_assistant.embeddings.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider integrating with Google Gemini via google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.gemini_embedding_api_key or settings.gemini_api_key
        self._model_name = model_name or settings.gemini_embedding_model or "gemini-embedding-001"
        self._dimension = dimension or 768

        self._client = None

    def _get_client(self):
        """Lazily initialize and return the google-genai Client."""
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required to generate embeddings. "
                    "Set the GEMINI_API_KEY environment variable or pass api_key to the provider, "
                    "or use the '--provider mock' option for offline testing."
                )
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "google-genai package is required. Install it with: pip install google-genai"
                ) from e
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @staticmethod
    def _normalize(vec: List[float]) -> List[float]:
        """Normalize vector to unit length (L2 norm = 1.0) so dot product == cosine similarity."""
        squared_sum = sum(x * x for x in vec)
        norm = math.sqrt(squared_sum)
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        results = self.embed_batch([text])
        return results[0]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embedding vectors for a list of strings in batches."""
        if not texts:
            return []

        client = self._get_client()
        from google.genai import types

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            # Replace empty strings with a space to prevent API rejection
            clean_chunk = [t if t.strip() else " " for t in chunk]

            config = None
            if self._dimension:
                config = types.EmbedContentConfig(output_dimensionality=self._dimension)

            try:
                response = client.models.embed_content(
                    model=self._model_name,
                    contents=clean_chunk,
                    config=config,
                )

                if hasattr(response, "embeddings") and response.embeddings:
                    for emb in response.embeddings:
                        vals = list(emb.values) if hasattr(emb, "values") else list(emb)
                        all_embeddings.append(self._normalize(vals))
                else:
                    logger.warning("No embeddings returned by Gemini API for batch starting at %d", i)
                    for _ in chunk:
                        all_embeddings.append([0.0] * self._dimension)
            except Exception as e:
                logger.error("Gemini embedding API call failed: %s", e)
                raise RuntimeError(f"Gemini embedding API error: {e}") from e

        return all_embeddings
