"""Unit tests for EmbeddingManager incremental synchronization."""

import tempfile
import unittest
from pathlib import Path

from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.ingestion.indexer import IncrementalIndexer


class TestEmbeddingManager(unittest.TestCase):
    """Test suite for EmbeddingManager."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_path = Path(self.temp_dir.name)
        self.cache_dir = self.repo_path / ".cache"

        (self.repo_path / "src").mkdir(parents=True)
        self.file1 = self.repo_path / "src" / "alpha.py"
        self.file2 = self.repo_path / "src" / "beta.py"

        self.file1.write_text("def alpha_function():\n    return 'alpha'\n", encoding="utf-8")
        self.file2.write_text("def beta_function():\n    return 'beta'\n", encoding="utf-8")

        self.indexer = IncrementalIndexer(self.repo_path, cache_dir=self.cache_dir)
        self.indexer.index()

        self.provider = MockEmbeddingProvider(dimension=64)
        self.manager = EmbeddingManager(
            repository_path=self.repo_path,
            cache_dir=self.cache_dir,
            provider=self.provider,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initial_sync_and_incremental_no_change(self):
        # 1. Initial sync
        report1 = self.manager.sync()
        self.assertGreaterEqual(report1["total_chunks"], 2)
        self.assertEqual(report1["embedded_chunks"], report1["total_chunks"])
        self.assertEqual(report1["evicted_chunks"], 0)
        self.assertEqual(self.manager.vector_store.count(), report1["total_chunks"])

        # 2. Second sync with no changes
        report2 = self.manager.sync()
        self.assertEqual(report2["embedded_chunks"], 0)
        self.assertEqual(report2["evicted_chunks"], 0)

    def test_incremental_file_modification(self):
        # First sync
        self.manager.sync()
        initial_count = self.manager.vector_store.count()

        # Modify alpha.py
        self.file1.write_text("def alpha_function_v2():\n    return 'new_alpha'\n", encoding="utf-8")
        self.indexer.index()

        # Incremental sync should only re-embed alpha's new chunk
        report = self.manager.sync()
        self.assertGreaterEqual(report["embedded_chunks"], 1)
        self.assertEqual(self.manager.vector_store.count(), initial_count)

    def test_file_deletion_evicts_vectors(self):
        self.manager.sync()
        initial_count = self.manager.vector_store.count()

        # Delete beta.py and re-index
        self.file2.unlink()
        self.indexer.index()

        # Sync should evict beta's chunk
        report = self.manager.sync()
        self.assertGreater(report["evicted_chunks"], 0)
        self.assertLess(self.manager.vector_store.count(), initial_count)

    def test_rebuild_sync(self):
        self.manager.sync()
        report = self.manager.sync(rebuild=True)
        self.assertEqual(report["embedded_chunks"], report["total_chunks"])


if __name__ == "__main__":
    unittest.main()
