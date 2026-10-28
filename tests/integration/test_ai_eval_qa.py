"""Integration test validating repository Q&A reasoning against read-only ai-evaluation."""

from pathlib import Path
import tempfile
import unittest

from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.ingestion.scanner import RepositoryScanner
from codebase_assistant.llm.mock_client import MockGeminiClient
from codebase_assistant.llm.qa import CodebaseQAEngine
from codebase_assistant.retrieval.hybrid import HybridRetriever


class TestAiEvalQAIntegration(unittest.TestCase):
    """End-to-end integration test answering repository questions over ai-evaluation."""

    @classmethod
    def setUpClass(cls):
        candidates = [
            Path("../ai_evaluation").resolve(),
            Path("../../ai_evaluation").resolve(),
            Path("/home/phaniranjan/AI/ai_evaluation").resolve(),
            Path(r"\\wsl.localhost\Ubuntu-26.04\home\phaniranjan\AI\ai_evaluation").resolve(),
        ]
        cls.ai_eval_dir = None
        for cand in candidates:
            if cand.is_dir() and (cand / "src" / "ai_evaluation").is_dir():
                cls.ai_eval_dir = cand
                break

    def setUp(self):
        if self.ai_eval_dir is None:
            self.skipTest("ai-evaluation repository not found on local path.")

        # Snapshot modification timestamps in target repo to assert strict read-only behavior
        scanner = RepositoryScanner(self.ai_eval_dir)
        self.eligible_files, _, _ = scanner.scan()
        self.initial_timestamps = {
            p: (self.ai_eval_dir / p).stat().st_mtime
            for p in self.eligible_files
            if (self.ai_eval_dir / p).is_file()
        }

        self.temp_cache = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_cache.name)

    def tearDown(self):
        self.temp_cache.cleanup()

        # Deterministic verification of strict read-only invariant
        current_timestamps = {
            p: (self.ai_eval_dir / p).stat().st_mtime
            for p in self.eligible_files
            if (self.ai_eval_dir / p).is_file()
        }
        self.assertEqual(
            self.initial_timestamps,
            current_timestamps,
            "Target repository was modified during Q&A test! Read-only invariant violated.",
        )

    def test_ai_eval_qa_answers_question_with_citations(self):
        # 1. Index repository into temporary cache
        indexer = IncrementalIndexer(self.ai_eval_dir, cache_dir=self.cache_dir)
        indexer.index()
        chunks = indexer.load_chunks()

        # 2. Embed with deterministic mock provider
        provider = MockEmbeddingProvider(dimension=256)
        manager = EmbeddingManager(
            repository_path=self.ai_eval_dir,
            cache_dir=self.cache_dir,
            provider=provider,
        )
        manager.sync()
        vector_store = manager.load_vector_store()

        # 3. Setup HybridRetriever and CodebaseQAEngine
        retriever = HybridRetriever(
            chunks=chunks,
            vector_store=vector_store,
            embedding_provider=provider,
        )
        llm_client = MockGeminiClient(model_name="mock-gemini-v1")
        qa_engine = CodebaseQAEngine(
            retriever=retriever,
            llm_client=llm_client,
            default_budget=8192,
        )

        # 4. Ask domain question about BM25 implementation
        query = "How is BM25Retriever implemented and what IDF scoring does it use?"
        response = qa_engine.ask(query=query, top_k=3, mode="hybrid")

        # 5. Verify answer content, citations, and metadata
        self.assertEqual(response.query, query)
        self.assertIn("rag_retriever.py", response.answer)
        self.assertGreaterEqual(len(response.citations), 1)

        citation_files = [c.file_path for c in response.citations]
        self.assertTrue(any("rag_retriever.py" in f for f in citation_files))
        self.assertGreater(response.assembled_chunks_count, 0)
        self.assertFalse(response.is_truncated)


if __name__ == "__main__":
    unittest.main()
