"""Vector storage and similarity search supporting pure Python, NumPy, and FAISS."""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from codebase_assistant.embeddings.base import BaseVectorStore

logger = logging.getLogger(__name__)


class VectorStore(BaseVectorStore):
    """Local vector store with cosine similarity nearest-neighbor search.

    Provides three execution tiers:
      Tier 1: FAISS (IndexFlatIP) when faiss and numpy are installed.
      Tier 2: NumPy matrix dot-product (np.dot) when numpy is installed.
      Tier 3: Pure-Python dot product for zero-dependency portability.

    All vectors are stored normalized so dot-product equals cosine similarity.
    Persists an embedding fingerprint (provider, model, dimension) to safeguard
    against mixed-vector spaces across model switches.
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        fingerprint: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.dimension = dimension
        self.fingerprint = fingerprint or {}
        self._vectors: Dict[str, List[float]] = {}
        self._metadatas: Dict[str, Dict[str, Any]] = {}

        # Acceleration flags
        self._numpy = None
        self._faiss = None
        self._try_load_accelerators()

    def _try_load_accelerators(self) -> None:
        """Attempt to lazily import numpy and faiss if available."""
        try:
            import numpy as np
            self._numpy = np
        except ImportError:
            self._numpy = None

        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            self._faiss = None

    @staticmethod
    def _normalize(vec: List[float]) -> List[float]:
        """Normalize vector to unit length (L2 norm = 1.0)."""
        squared_sum = sum(x * x for x in vec)
        norm = math.sqrt(squared_sum)
        if norm == 0.0:
            return vec
        return [x / norm for x in vec]

    def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add or update vectors in the store."""
        if len(ids) != len(vectors):
            raise ValueError(f"ids count ({len(ids)}) does not match vectors count ({len(vectors)})")

        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas count does not match ids count")

        for idx, (chunk_id, vec) in enumerate(zip(ids, vectors)):
            if self.dimension is None:
                self.dimension = len(vec)
            elif len(vec) != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch for '{chunk_id}': expected {self.dimension}, got {len(vec)}"
                )

            norm_vec = self._normalize(vec)
            self._vectors[chunk_id] = norm_vec

            if metadatas and idx < len(metadatas):
                self._metadatas[chunk_id] = metadatas[idx]

    def remove(self, ids: List[str]) -> None:
        """Remove specified chunk IDs from the vector store."""
        for chunk_id in ids:
            self._vectors.pop(chunk_id, None)
            self._metadatas.pop(chunk_id, None)

    def count(self) -> int:
        """Return total number of stored vectors."""
        return len(self._vectors)

    def get_vector(self, chunk_id: str) -> Optional[List[float]]:
        """Retrieve stored vector for a chunk ID."""
        return self._vectors.get(chunk_id)

    def get_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored metadata for a chunk ID."""
        return self._metadatas.get(chunk_id)

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the top_k most similar vectors to query_vector using cosine similarity."""
        if not self._vectors:
            return []

        if top_k <= 0:
            return []

        norm_query = self._normalize(query_vector)

        # Tier 1: FAISS acceleration if faiss and numpy are available
        if self._faiss is not None and self._numpy is not None and len(self._vectors) > 20:
            try:
                return self._search_faiss(norm_query, top_k)
            except Exception as e:
                logger.warning("FAISS search failed, falling back to NumPy/pure-Python: %s", e)

        # Tier 2: NumPy matrix dot-product acceleration
        if self._numpy is not None and len(self._vectors) > 20:
            return self._search_numpy(norm_query, top_k)

        # Tier 3: Pure-Python dot product
        scores: List[Tuple[str, float]] = []
        for chunk_id, vec in self._vectors.items():
            # Dot product of normalized vectors == cosine similarity
            sim = sum(q * v for q, v in zip(norm_query, vec))
            scores.append((chunk_id, float(sim)))

        # Sort descending by similarity
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _search_faiss(self, query: List[float], top_k: int) -> List[Tuple[str, float]]:
        """FAISS IndexFlatIP accelerated inner product search."""
        np = self._numpy
        ids = list(self._vectors.keys())
        matrix = np.array([self._vectors[cid] for cid in ids], dtype=np.float32)
        dim = matrix.shape[1]
        index = self._faiss.IndexFlatIP(dim)
        index.add(matrix)
        q = np.array([query], dtype=np.float32)
        k = min(top_k, len(ids))
        distances, indices = index.search(q, k)
        return [(ids[idx], float(dist)) for dist, idx in zip(distances[0], indices[0]) if idx >= 0]

    def _search_numpy(self, query: List[float], top_k: int) -> List[Tuple[str, float]]:
        """NumPy matrix dot-product acceleration."""
        np = self._numpy
        ids = list(self._vectors.keys())
        matrix = np.array([self._vectors[cid] for cid in ids], dtype=np.float32)
        q = np.array(query, dtype=np.float32)
        sims = np.dot(matrix, q)

        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(ids[idx], float(sims[idx])) for idx in top_indices]

    def save(self, path: Path) -> None:
        """Atomically persist vector store to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")

        payload = {
            "dimension": self.dimension,
            "fingerprint": self.fingerprint,
            "count": len(self._vectors),
            "vectors": self._vectors,
            "metadatas": self._metadatas,
        }

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(str(temp_file), str(target_path))
        except BaseException:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass
            raise

    def load(self, path: Path) -> None:
        """Load vector store from disk."""
        target_path = Path(path)
        if not target_path.is_file():
            logger.warning("Vector store file %s does not exist.", target_path)
            return

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.dimension = data.get("dimension")
        self.fingerprint = data.get("fingerprint", {})
        self._vectors = data.get("vectors", {})
        self._metadatas = data.get("metadatas", {})

    def clear(self) -> None:
        """Clear all stored vectors and metadata."""
        self._vectors.clear()
        self._metadatas.clear()
