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

    def test_renamed_file_evicts_old_and_indexes_new(self):
        indexer = IncrementalIndexer(self.repo_dir, cache_dir=self.cache_dir)
        indexer.index()

        # Rename beta.py -> beta_renamed.py
        new_beta = self.repo_dir / "src" / "beta_renamed.py"
        new_beta.write_text(self.file_b.read_text(encoding="utf-8"), encoding="utf-8")
        self.file_b.unlink()

        report = indexer.index()
        self.assertIn("src/beta.py", report.deleted_files)
        self.assertIn("src/beta_renamed.py", report.added_files)

        chunks = indexer.load_chunks()
        old_chunks = [c for c in chunks.values() if c.file_path == "src/beta.py"]
        new_chunks = [c for c in chunks.values() if c.file_path == "src/beta_renamed.py"]

        self.assertEqual(len(old_chunks), 0)
        self.assertTrue(len(new_chunks) >= 1)

    def test_empty_file_handling(self):
        indexer = IncrementalIndexer(self.repo_dir, cache_dir=self.cache_dir)
        indexer.index()

        # 1. Add new empty file
        empty_file = self.repo_dir / "src" / "empty.py"
        empty_file.write_text("", encoding="utf-8")

        report1 = indexer.index()
        self.assertIn("src/empty.py", report1.added_files)

        manifest = indexer.load_manifest()
        self.assertIn("src/empty.py", manifest.files)
        self.assertEqual(manifest.files["src/empty.py"].chunk_ids, [])

        # 2. Modify alpha.py to be empty: verify old chunks are purged
        self.file_a.write_text("   \n\n  ", encoding="utf-8")
        report2 = indexer.index()
        self.assertIn("src/alpha.py", report2.modified_files)

        chunks = indexer.load_chunks()
        alpha_chunks = [c for c in chunks.values() if c.file_path == "src/alpha.py"]
        self.assertEqual(len(alpha_chunks), 0)

    def test_parsing_failure_does_not_leave_stale_chunks(self):
        indexer = IncrementalIndexer(self.repo_dir, cache_dir=self.cache_dir)
        indexer.index()

        # Mock chunker to simulate unexpected parser failure
        original_chunk_file = indexer.chunker.chunk_file

        def failing_chunk_file(content, rel_file_path, language, source_revision=None):
            if "alpha.py" in rel_file_path:
                raise RuntimeError("Simulated unrecoverable AST parser crash")
            return original_chunk_file(content, rel_file_path, language, source_revision)

        indexer.chunker.chunk_file = failing_chunk_file

        # Modify alpha.py
        self.file_a.write_text("def modified_during_crash():\n    pass\n", encoding="utf-8")

        report = indexer.index()
        self.assertIn("src/alpha.py", report.modified_files)

        # Ensure old alpha chunks were purged and no stale chunks appear current
        chunks = indexer.load_chunks()
        alpha_chunks = [c for c in chunks.values() if c.file_path == "src/alpha.py"]
        self.assertEqual(len(alpha_chunks), 0)

        manifest = indexer.load_manifest()
        self.assertEqual(manifest.files["src/alpha.py"].chunk_ids, [])

    def test_repository_identity_and_revision_verification(self):
        indexer = IncrementalIndexer(self.repo_dir, cache_dir=self.cache_dir)
        indexer.index()

        manifest = indexer.load_manifest()
        self.assertIsNotNone(manifest)
        self.assertTrue(len(manifest.repository_id) > 0)
        self.assertEqual(manifest.repository_path, self.repo_dir.as_posix())

        # Attempt to load this manifest using an indexer targeting a DIFFERENT repo
        other_repo = tempfile.TemporaryDirectory()
        mismatched_indexer = IncrementalIndexer(Path(other_repo.name), cache_dir=self.cache_dir)

        # Must detect repository identity mismatch and refuse to reuse the cache
        self.assertIsNone(mismatched_indexer.load_manifest())
        other_repo.cleanup()


if __name__ == "__main__":
    unittest.main()
