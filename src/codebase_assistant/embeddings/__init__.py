"""Embeddings and vector storage module."""

from codebase_assistant.embeddings.base import BaseEmbeddingProvider, BaseVectorStore
from codebase_assistant.embeddings.gemini import GeminiEmbeddingProvider
from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.embeddings.vector_store import VectorStore

__all__ = [
    "BaseEmbeddingProvider",
    "BaseVectorStore",
    "GeminiEmbeddingProvider",
    "MockEmbeddingProvider",
    "VectorStore",
    "EmbeddingManager",
]
