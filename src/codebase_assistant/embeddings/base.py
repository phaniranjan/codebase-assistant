"""Base abstractions for embedding providers and vector storage."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class BaseEmbeddingProvider(ABC):
    """Abstract base class for code and text embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality of this embedding model."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name or identifier of the embedding model."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate an embedding vector for a single piece of text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of text pieces."""
        pass


class BaseVectorStore(ABC):
    """Abstract interface for local vector storage and nearest-neighbor search."""

    @abstractmethod
    def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add or update vectors in the index with corresponding IDs and optional metadata."""
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Search the index for the top-k nearest vectors using cosine similarity.

        Returns a list of (id, similarity_score) tuples sorted in descending order of similarity.
        """
        pass

    @abstractmethod
    def remove(self, ids: List[str]) -> None:
        """Remove specified IDs from the vector index."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize index and metadata to disk."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load index and metadata from disk."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of vectors in the store."""
        pass
