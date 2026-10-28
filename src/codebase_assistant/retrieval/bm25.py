"""Okapi BM25 sparse keyword retriever tailored for source code."""

from collections import Counter, defaultdict
import math
from typing import Dict, List, Optional, Tuple

from codebase_assistant.ingestion.models import CodeChunk
from codebase_assistant.retrieval.tokenizer import CodeTokenizer


class BM25Retriever:
    """In-memory Okapi BM25 retriever with code-symbol and path boosting."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        symbol_boost: float = 2.5,
        path_boost: float = 2.0,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.symbol_boost = symbol_boost
        self.path_boost = path_boost

        self.doc_count: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: Dict[str, float] = {}

        # Inverted index: term -> {doc_id: weighted_frequency}
        self.inverted_index: Dict[str, Dict[str, float]] = defaultdict(dict)
        # IDF cache: term -> idf
        self.idf_cache: Dict[str, float] = {}
        # Chunks store: chunk_id -> CodeChunk
        self.chunks: Dict[str, CodeChunk] = {}

    def index(self, chunks: List[CodeChunk]) -> None:
        """Build the BM25 inverted index from a list of CodeChunks."""
        self.inverted_index.clear()
        self.idf_cache.clear()
        self.doc_lengths.clear()
        self.chunks.clear()

        self.doc_count = len(chunks)
        if self.doc_count == 0:
            self.avg_doc_len = 0.0
            return

        total_length: float = 0.0

        for chunk in chunks:
            cid = chunk.chunk_id
            self.chunks[cid] = chunk

            # 1. Tokenize content (weight 1.0)
            content_tokens = CodeTokenizer.tokenize(chunk.content)
            # 2. Tokenize symbol name (weight symbol_boost)
            symbol_tokens = CodeTokenizer.tokenize(chunk.symbol_name) if chunk.symbol_name else []
            # 3. Tokenize file path (weight path_boost)
            path_tokens = CodeTokenizer.tokenize(chunk.file_path)

            weighted_counts: Dict[str, float] = defaultdict(float)

            for t in content_tokens:
                weighted_counts[t] += 1.0
            for t in symbol_tokens:
                weighted_counts[t] += self.symbol_boost
            for t in path_tokens:
                weighted_counts[t] += self.path_boost

            doc_len = sum(weighted_counts.values())
            self.doc_lengths[cid] = doc_len
            total_length += doc_len

            for term, freq in weighted_counts.items():
                self.inverted_index[term][cid] = freq

        self.avg_doc_len = total_length / self.doc_count if self.doc_count > 0 else 0.0

        # Precompute IDF for all terms in the vocabulary
        for term, postings in self.inverted_index.items():
            n = len(postings)
            # Standard Lucene/BM25 IDF formula: ln(1 + (N - n + 0.5) / (n + 0.5))
            idf = math.log(1.0 + (self.doc_count - n + 0.5) / (n + 0.5))
            self.idf_cache[term] = max(idf, 0.0)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search indexed chunks using Okapi BM25.

        Returns list of (chunk_id, bm25_score) sorted descending.
        """
        if self.doc_count == 0 or not query.strip() or top_k <= 0:
            return []

        query_tokens = CodeTokenizer.tokenize(query)
        if not query_tokens:
            return []

        # Count term frequencies in query
        q_counts = Counter(query_tokens)
        scores: Dict[str, float] = defaultdict(float)

        for term, _ in q_counts.items():
            if term not in self.inverted_index:
                continue

            idf = self.idf_cache.get(term, 0.0)
            if idf <= 0.0:
                continue

            postings = self.inverted_index[term]
            for doc_id, tf in postings.items():
                doc_len = self.doc_lengths.get(doc_id, self.avg_doc_len)
                # BM25 term score component
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                scores[doc_id] += idf * (numerator / denominator)

        # Sort by score descending
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
