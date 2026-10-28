"""Hybrid retriever combining dense vector search and sparse BM25 with Reciprocal Rank Fusion."""

from typing import Dict, List, Literal, Optional, Set, Tuple

from pydantic import BaseModel, Field

from codebase_assistant.embeddings.base import BaseEmbeddingProvider
from codebase_assistant.embeddings.vector_store import VectorStore
from codebase_assistant.ingestion.models import CodeChunk
from codebase_assistant.retrieval.bm25 import BM25Retriever


class SearchResult(BaseModel):
    """Encapsulates a retrieved code chunk with ranking scores."""

    chunk_id: str
    chunk: CodeChunk
    score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


class HybridRetriever:
    """Hybrid code retriever fusing dense vector search and sparse BM25 retrieval."""

    def __init__(
        self,
        chunks: Dict[str, CodeChunk],
        vector_store: VectorStore,
        embedding_provider: BaseEmbeddingProvider,
        bm25: Optional[BM25Retriever] = None,
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> None:
        self.chunks = chunks
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        if bm25 is not None:
            self.bm25 = bm25
        else:
            self.bm25 = BM25Retriever()
            self.bm25.index(list(chunks.values()))

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["hybrid", "dense_only", "sparse_only"] = "hybrid",
    ) -> List[SearchResult]:
        """Execute retrieval using hybrid fusion, dense-only, or sparse-only mode."""
        if not query.strip() or top_k <= 0 or not self.chunks:
            return []

        if mode == "sparse_only":
            return self._search_sparse_only(query, top_k)
        elif mode == "dense_only":
            return self._search_dense_only(query, top_k)
        elif mode == "hybrid":
            return self._search_hybrid(query, top_k)
        else:
            raise ValueError(f"Unknown search mode: '{mode}'. Expected 'hybrid', 'dense_only', or 'sparse_only'.")

    def _search_sparse_only(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform sparse BM25 keyword search."""
        raw_results = self.bm25.search(query, top_k=top_k)
        results: List[SearchResult] = []
        for rank, (cid, score) in enumerate(raw_results, start=1):
            chunk = self.chunks.get(cid)
            if chunk:
                results.append(
                    SearchResult(
                        chunk_id=cid,
                        chunk=chunk,
                        score=score,
                        sparse_rank=rank,
                        sparse_score=score,
                    )
                )
        return results

    def _search_dense_only(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform dense embedding vector search."""
        query_vector = self.embedding_provider.embed_text(query)
        raw_results = self.vector_store.search(query_vector, top_k=top_k)
        results: List[SearchResult] = []
        for rank, (cid, score) in enumerate(raw_results, start=1):
            chunk = self.chunks.get(cid)
            if chunk:
                results.append(
                    SearchResult(
                        chunk_id=cid,
                        chunk=chunk,
                        score=score,
                        dense_rank=rank,
                        dense_score=score,
                    )
                )
        return results

    def _search_hybrid(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform Reciprocal Rank Fusion (RRF) between dense and sparse results."""
        # Retrieve candidate pools (up to 3x top_k to provide solid overlap)
        candidate_k = max(top_k * 3, 20)

        # 1. Sparse BM25
        sparse_hits = self.bm25.search(query, top_k=candidate_k)
        sparse_ranks: Dict[str, int] = {cid: rank for rank, (cid, _) in enumerate(sparse_hits, start=1)}
        sparse_scores: Dict[str, float] = {cid: score for cid, score in sparse_hits}

        # 2. Dense Vector Search
        query_vec = self.embedding_provider.embed_text(query)
        dense_hits = self.vector_store.search(query_vec, top_k=candidate_k)
        dense_ranks: Dict[str, int] = {cid: rank for rank, (cid, _) in enumerate(dense_hits, start=1)}
        dense_scores: Dict[str, float] = {cid: score for cid, score in dense_hits}

        all_candidate_ids: Set[str] = set(sparse_ranks.keys()) | set(dense_ranks.keys())

        # 3. Calculate Reciprocal Rank Fusion scores
        rrf_scores: Dict[str, float] = {}
        for cid in all_candidate_ids:
            score = 0.0
            if cid in dense_ranks:
                score += self.dense_weight * (1.0 / (self.rrf_k + dense_ranks[cid]))
            if cid in sparse_ranks:
                score += self.sparse_weight * (1.0 / (self.rrf_k + sparse_ranks[cid]))
            rrf_scores[cid] = score

        # 4. Sort candidates by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        results: List[SearchResult] = []
        for cid in sorted_cids[:top_k]:
            chunk = self.chunks.get(cid)
            if chunk:
                results.append(
                    SearchResult(
                        chunk_id=cid,
                        chunk=chunk,
                        score=rrf_scores[cid],
                        dense_rank=dense_ranks.get(cid),
                        sparse_rank=sparse_ranks.get(cid),
                        dense_score=dense_scores.get(cid),
                        sparse_score=sparse_scores.get(cid),
                    )
                )

        return results
