"""Integration test verifying read-only scanning and ingestion of the ai-evaluation repository."""

from pathlib import Path
import tempfile
import unittest

from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.ingestion.scanner import RepositoryScanner


class TestAiEvalIngestion(unittest.TestCase):
    """Test ingestion against real ai-evaluation codebase without modifying it."""

    @classmethod
    def setUpClass(cls):
        # Locate ai-evaluation repository relative to this workspace
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
        self.temp_cache = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_cache.name)

    def tearDown(self):
        self.temp_cache.cleanup()

    def test_scan_ai_eval_filters_secrets_and_finds_python(self):
        scanner = RepositoryScanner(self.ai_eval_dir)
        eligible, revision, excluded = scanner.scan()

        eligible_str = [p.as_posix() for p in eligible]

        # 1. Verify key application and test files are found
        self.assertIn("src/ai_evaluation/rag_pipeline.py", eligible_str)
        self.assertIn("src/ai_evaluation/rag_retriever.py", eligible_str)
        self.assertIn("src/ai_evaluation/llm_generator.py", eligible_str)
        self.assertIn("src/ai_evaluation/agents/minimal_agent.py", eligible_str)
        self.assertIn("README.md", eligible_str)
        self.assertIn("pyproject.toml", eligible_str)

        # 2. Verify secrets and virtual environments in ai-evaluation are strictly excluded
        self.assertNotIn(".env", eligible_str)
        self.assertNotIn(".env.local", eligible_str)
        self.assertTrue(all(not p.startswith("deepeval_venv") for p in eligible_str))
        self.assertTrue(all(not p.startswith(".git") for p in eligible_str))
        self.assertTrue(all(not "__pycache__" in p for p in eligible_str))

        # 3. Verify revision is detected
        self.assertIsNotNone(revision)
        self.assertNotEqual(revision.revision_id, "unknown")

    def test_ingest_ai_eval_extracts_rich_ast_chunks_without_modifications(self):
        # Snapshot modification times of key files before indexing
        pipeline_file = self.ai_eval_dir / "src" / "ai_evaluation" / "rag_pipeline.py"
        mtime_before = pipeline_file.stat().st_mtime

        indexer = IncrementalIndexer(self.ai_eval_dir, cache_dir=self.cache_dir)
        report = indexer.index()

        # Check indexing report
        self.assertTrue(report.total_files > 10)
        self.assertTrue(report.total_chunks > 30)

        chunks = indexer.load_chunks()
        symbol_names = {c.symbol_name for c in chunks.values() if c.symbol_name}

        # Check symbol extraction
        self.assertIn("RAGPipeline", symbol_names)
        self.assertIn("RAGPipeline.query", symbol_names)
        self.assertIn("BM25Retriever", symbol_names)
        self.assertIn("BM25Retriever.retrieve_chunks", symbol_names)
        self.assertIn("LLMGenerator", symbol_names)
        self.assertIn("MinimalAgent", symbol_names)

        # Verify that ai-evaluation was strictly treated as read-only
        mtime_after = pipeline_file.stat().st_mtime
        self.assertEqual(mtime_before, mtime_after)


if __name__ == "__main__":
    unittest.main()
