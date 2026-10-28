"""Unit tests for HybridRetriever with Reciprocal Rank Fusion."""

import unittest

from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.embeddings.vector_store import VectorStore
from codebase_assistant.ingestion.models import ChunkType, CodeChunk
from codebase_assistant.retrieval.hybrid import HybridRetriever


class TestHybridRetriever(unittest.TestCase):
    """Test suite for HybridRetriever."""

    def setUp(self):
        self.chunk_a = CodeChunk(
            chunk_id="chunk_auth",
            file_path="src/auth.py",
            content="def authenticate_user(token: str):\n    return verify_jwt(token)",
            symbol_name="authenticate_user",
            language="python",
            chunk_type=ChunkType.FUNCTION,
            start_line=1,
            end_line=2,
            content_hash="h_auth",
        )
        self.chunk_b = CodeChunk(
            chunk_id="chunk_db",
            file_path="src/database.py",
            content="class DatabaseConnection:\n    def connect(self): pass",
            symbol_name="DatabaseConnection",
            language="python",
            chunk_type=ChunkType.CLASS,
            start_line=1,
            end_line=2,
            content_hash="h_db",
        )

        self.chunks = {
            self.chunk_a.chunk_id: self.chunk_a,
            self.chunk_b.chunk_id: self.chunk_b,
        }

        self.provider = MockEmbeddingProvider(dimension=128)
        self.vector_store = VectorStore(dimension=128)

        # Embed and store
        vec_a = self.provider.embed_text(self.chunk_a.content)
        vec_b = self.provider.embed_text(self.chunk_b.content)
        self.vector_store.add(["chunk_auth", "chunk_db"], [vec_a, vec_b])

        self.retriever = HybridRetriever(
            chunks=self.chunks,
            vector_store=self.vector_store,
            embedding_provider=self.provider,
        )

    def test_hybrid_search_rrf(self):
        results = self.retriever.search("authenticate user token", top_k=2, mode="hybrid")
        self.assertEqual(len(results), 2)
        # Auth chunk should rank top
        self.assertEqual(results[0].chunk_id, "chunk_auth")
        self.assertIsNotNone(results[0].dense_rank)
        self.assertIsNotNone(results[0].sparse_rank)
        self.assertGreater(results[0].score, results[1].score)

    def test_dense_only_search(self):
        results = self.retriever.search("database connection", top_k=1, mode="dense_only")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk_db")
        self.assertIsNotNone(results[0].dense_score)
        self.assertIsNone(results[0].sparse_score)

    def test_sparse_only_search(self):
        results = self.retriever.search("verify_jwt", top_k=1, mode="sparse_only")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "chunk_auth")
        self.assertIsNotNone(results[0].sparse_score)
        self.assertIsNone(results[0].dense_score)

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self.retriever.search("query", mode="unknown_mode")


if __name__ == "__main__":
    unittest.main()
