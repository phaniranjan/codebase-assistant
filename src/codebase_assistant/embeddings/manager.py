"""Coordinates chunk embedding generation, incremental caching, and vector storage."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from codebase_assistant.config import get_settings
from codebase_assistant.embeddings.base import BaseEmbeddingProvider
from codebase_assistant.embeddings.gemini import GeminiEmbeddingProvider
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.embeddings.vector_store import VectorStore
from codebase_assistant.ingestion.indexer import IncrementalIndexer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding generation and synchronizes the local vector store with indexed chunks."""

    def __init__(
        self,
        repository_path: Path | str,
        cache_dir: Optional[Path | str] = None,
        provider: Optional[BaseEmbeddingProvider] = None,
    ) -> None:
        self.repository_path = Path(repository_path).resolve()
        settings = get_settings()

        if cache_dir:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            self.cache_dir = self.repository_path / settings.cache_dir

        self.vector_store_path = self.cache_dir / "vector_index.json"
        self.indexer = IncrementalIndexer(self.repository_path, cache_dir=self.cache_dir)

        if provider is not None:
            self.provider = provider
        else:
            # Default to Gemini if API key is configured, otherwise fallback to Mock
            if settings.gemini_embedding_api_key or settings.gemini_api_key:
                try:
                    self.provider = GeminiEmbeddingProvider()
                except Exception as e:
                    logger.warning("Could not initialize GeminiEmbeddingProvider: %s. Using MockEmbeddingProvider.", e)
                    self.provider = MockEmbeddingProvider()
            else:
                self.provider = MockEmbeddingProvider()

        self.vector_store = VectorStore(dimension=self.provider.dimension)

    def load_vector_store(self) -> VectorStore:
        """Load vector store from cache if present."""
        if self.vector_store_path.is_file():
            self.vector_store.load(self.vector_store_path)
        return self.vector_store

    def sync(self, rebuild: bool = False, batch_size: int = 32) -> Dict[str, Any]:
        """Synchronize vector store with current chunks on disk incrementally.

        Only generates embeddings for newly added or modified chunks; evicts chunks
        from deleted files or older revisions.
        """
        chunks = self.indexer.load_chunks()
        if not chunks:
            logger.warning("No chunks found in cache for %s. Run indexing first.", self.repository_path)
            return {
                "total_chunks": 0,
                "embedded_chunks": 0,
                "evicted_chunks": 0,
                "dimension": self.provider.dimension,
                "model": self.provider.model_name,
            }

        if rebuild:
            self.vector_store.clear()
        else:
            self.load_vector_store()

        current_chunk_ids = set(chunks.keys())
        stored_ids = set(self.vector_store._vectors.keys())

        # 1. Stale chunks eviction
        stale_ids = list(stored_ids - current_chunk_ids)
        if stale_ids:
            self.vector_store.remove(stale_ids)
            logger.info("Evicted %d stale vectors from vector store.", len(stale_ids))

        # 2. Identify missing or updated chunks
        ids_to_embed: List[str] = []
        texts_to_embed: List[str] = []
        metadatas_to_embed: List[Dict[str, Any]] = []

        for cid, chunk in chunks.items():
            stored_meta = self.vector_store.get_metadata(cid)
            # Re-embed if not in vector store or if content hash changed
            if stored_meta is None or stored_meta.get("content_hash") != chunk.content_hash:
                ids_to_embed.append(cid)
                # Form text to embed: prepend symbol name and file path for enhanced semantic context
                prefix = f"File: {chunk.file_path}\nSymbol: {chunk.symbol_name}\n\n"
                texts_to_embed.append(prefix + chunk.content)
                metadatas_to_embed.append({
                    "chunk_id": cid,
                    "file_path": chunk.file_path,
                    "symbol_name": chunk.symbol_name,
                    "chunk_type": chunk.chunk_type.value,
                    "language": chunk.language,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content_hash": chunk.content_hash,
                })

        # 3. Generate embeddings in batches
        if ids_to_embed:
            logger.info("Generating embeddings for %d chunks using %s...", len(ids_to_embed), self.provider.model_name)
            for i in range(0, len(ids_to_embed), batch_size):
                batch_ids = ids_to_embed[i : i + batch_size]
                batch_texts = texts_to_embed[i : i + batch_size]
                batch_meta = metadatas_to_embed[i : i + batch_size]

                batch_vectors = self.provider.embed_batch(batch_texts)
                self.vector_store.add(batch_ids, batch_vectors, batch_meta)

        # 4. Save vector store
        self.vector_store.save(self.vector_store_path)

        return {
            "total_chunks": len(chunks),
            "embedded_chunks": len(ids_to_embed),
            "evicted_chunks": len(stale_ids),
            "dimension": self.provider.dimension,
            "model": self.provider.model_name,
        }
