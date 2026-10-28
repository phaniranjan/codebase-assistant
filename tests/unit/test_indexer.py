"""Unit tests for incremental indexer and change detection."""

from pathlib import Path
import tempfile
import unittest

from codebase_assistant.ingestion.indexer import IncrementalIndexer


class TestIncrementalIndexer(unittest.TestCase):
    """Test incremental indexing, change detection, and stale chunk eviction."""

    def setUp(self):
        self.temp_repo = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.temp_repo.name)
        self.cache_dir = self.repo_dir / ".cache_test"

        # Create initial repository files
        (self.repo_dir / "src").mkdir(parents=True)
        self.file_a = self.repo_dir / "src" / "alpha.py"
        self.file_a.write_text("def alpha():\n    return 'alpha'\n", encoding="utf-8")

        self.file_b = self.repo_dir / "src" / "beta.py"
        self.file_b.write_text("def beta():\n    return 'beta'\n", encoding="utf-8")

    def tearDown(self):
        self.temp_repo.cleanup()

    def test_initial_indexing(self):
        indexer = IncrementalIndexer(self.repo_dir, cache_dir=self.cache_dir)
        report = indexer.index()

        self.assertEqual(len(report.added_files), 2)
        self.assertEqual(len(report.modified_files), 0)
        self.assertEqual(len(report.deleted_files), 0)
        self.assertEqual(report.total_files, 2)
        self.assertTrue(report.total_chunks >= 2)

        # Check files on disk
        manifest = indexer.load_manifest()
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.file_count, 2)

        chunks = indexer.load_chunks()
        self.assertEqual(len(chunks), report.total_chunks)

    def test_incremental_no_changes(self):
        indexer = IncrementalIndexer(self.repo_dir, cache_dir=self.cache_dir)
        report1 = indexer.index()
        self.assertEqual(len(report1.added_files), 2)

        # Run index again with zero modifications
        report2 = indexer.index()
        self.assertTrue(report2.is_fresh())
        self.assertEqual(len(report2.added_files), 0)
        self.assertEqual(len(report2.modified_files), 0)
        self.assertEqual(len(report2.deleted_files), 0)
        self.assertEqual(len(report2.unchanged_files), 2)

    def test_incremental_file_modification(self):
        indexer = IncrementalIndexer(self.repo_dir, cache_dir=self.cache_dir)
        indexer.index()

        # Modify alpha.py
        self.file_a.write_text("def alpha_v2():\n    return 'updated'\n", encoding="utf-8")

        report = indexer.index()
        self.assertEqual(report.modified_files, ["src/alpha.py"])
        self.assertEqual(len(report.added_files), 0)
        self.assertEqual(len(report.deleted_files), 0)

        # Verify old chunk was replaced
        chunks = indexer.load_chunks()
        func_symbols = [
            c.symbol_name for c in chunks.values()
            if c.file_path == "src/alpha.py" and c.chunk_type.value == "function"
        ]
        self.assertIn("alpha_v2", func_symbols)
        self.assertNotIn("alpha", func_symbols)

    def test_incremental_file_deletion_and_stale_eviction(self):
        indexer = IncrementalIndexer(self.repo_dir, cache_dir=self.cache_dir)
        indexer.index()

        # Delete beta.py
        self.file_b.unlink()

        report = indexer.index()
        self.assertEqual(report.deleted_files, ["src/beta.py"])
        self.assertEqual(report.total_files, 1)

        # Verify beta chunks were completely evicted
        chunks = indexer.load_chunks()
        beta_chunks = [c for c in chunks.values() if c.file_path == "src/beta.py"]
        self.assertEqual(len(beta_chunks), 0)

    def test_rebuild_index(self):
        indexer = IncrementalIndexer(self.repo_dir, cache_dir=self.cache_dir)
        indexer.index()

        report = indexer.index(rebuild=True)
        self.assertTrue(report.rebuilt)
        self.assertEqual(len(report.added_files), 2)
        self.assertEqual(report.total_files, 2)


if __name__ == "__main__":
    unittest.main()
