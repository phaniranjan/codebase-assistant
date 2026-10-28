"""Integration test validating hybrid retrieval against the read-only ai-evaluation repository."""

from pathlib import Path
import tempfile
import unittest

from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.ingestion.scanner import RepositoryScanner
from codebase_assistant.retrieval.context_assembler import ContextAssembler
from codebase_assistant.retrieval.hybrid import HybridRetriever


class TestAiEvalRetrievalIntegration(unittest.TestCase):
    """End-to-end integration test for indexing and hybrid retrieval on ai-evaluation."""

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

        # Fast snapshot of eligible files using pruned scanner (avoids traversing deepeval_venv)
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
            "Target repository was modified during retrieval test! Read-only invariant violated.",
        )

    def test_ai_eval_hybrid_retrieval_finds_rag_components(self):
        # 1. Index the repository into temporary cache
        indexer = IncrementalIndexer(self.ai_eval_dir, cache_dir=self.cache_dir)
        index_report = indexer.index()
        self.assertGreater(index_report.total_files, 0)
        self.assertGreater(index_report.total_chunks, 0)

        # 2. Synchronize embeddings using deterministic MockEmbeddingProvider
        provider = MockEmbeddingProvider(dimension=256)
        manager = EmbeddingManager(
            repository_path=self.ai_eval_dir,
            cache_dir=self.cache_dir,
            provider=provider,
        )
        sync_report = manager.sync()
        self.assertEqual(sync_report["embedded_chunks"], index_report.total_chunks)

        # 3. Initialize HybridRetriever
        chunks = indexer.load_chunks()
        vector_store = manager.load_vector_store()
        retriever = HybridRetriever(
            chunks=chunks,
            vector_store=vector_store,
            embedding_provider=provider,
        )

        # 4. Query 1: BM25Retriever implementation
        results_bm25 = retriever.search("BM25Retriever Robertson Spärck Jones IDF", top_k=3, mode="hybrid")
        self.assertGreaterEqual(len(results_bm25), 1)
        top_hit_bm25 = results_bm25[0]
        self.assertIn("rag_retriever.py", top_hit_bm25.chunk.file_path)

        # 5. Query 2: RAGPipeline implementation
        results_rag = retriever.search("class RAGPipeline generator grounding prompt", top_k=3, mode="hybrid")
        self.assertGreaterEqual(len(results_rag), 1)
        rag_files = [r.chunk.file_path for r in results_rag]
        self.assertTrue(
            any("rag_pipeline.py" in f for f in rag_files),
            f"rag_pipeline.py not found in retrieved results: {rag_files}",
        )

        # 6. Assemble context block with budget
        assembler = ContextAssembler(default_max_tokens=4000)
        context = assembler.assemble(results_rag, max_tokens=4000)
        self.assertIn("rag_pipeline.py", context.formatted_text)
        self.assertIn("```python", context.formatted_text)
        self.assertGreater(context.estimated_tokens, 10)


if __name__ == "__main__":
    unittest.main()
