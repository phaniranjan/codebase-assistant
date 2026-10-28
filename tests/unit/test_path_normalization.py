"""Unit tests for cross-platform path normalization, atomic cache writes, and failure visibility."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codebase_assistant.ingestion.chunker import LogicalChunker
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.ingestion.scanner import RepositoryScanner


class TestPathNormalizationAndHardening(unittest.TestCase):
    """Test path normalization, spaces in paths, atomic cache protection, and error visibility."""

    def test_chunk_id_normalization_across_path_separators(self):
        # A file path with Windows backslashes must yield the EXACT same ID as forward slashes
        posix_path = "src/modules/pipeline.py"
        windows_path = r"src\modules\pipeline.py"

        id_posix = LogicalChunker.generate_chunk_id(posix_path, "process_data", 25)
        id_windows = LogicalChunker.generate_chunk_id(windows_path, "process_data", 25)

        self.assertEqual(id_posix, id_windows)
        self.assertNotIn("\\", id_windows)

    def test_repository_path_containing_spaces(self):
        # Repositories checked out to folders with spaces (e.g. C:/My Projects/app space)
        temp_dir = tempfile.TemporaryDirectory(prefix="my repo with spaces ")
        repo_path = Path(temp_dir.name)
        cache_dir = repo_path / ".cache space"

        try:
            pkg_dir = repo_path / "src" / "sub module"
            pkg_dir.mkdir(parents=True)
            source_file = pkg_dir / "my worker.py"
            source_file.write_text("def worker_task():\n    return 'done'\n", encoding="utf-8")

            scanner = RepositoryScanner(repo_path)
            eligible, _, _ = scanner.scan()

            self.assertEqual(len(eligible), 1)
            self.assertEqual(eligible[0].as_posix(), "src/sub module/my worker.py")

            indexer = IncrementalIndexer(repo_path, cache_dir=cache_dir)
            report = indexer.index()

            self.assertEqual(len(report.added_files), 1)
            self.assertEqual(report.added_files[0], "src/sub module/my worker.py")
            self.assertEqual(len(report.failed_files), 0)

            # Check chunks
            chunks = indexer.load_chunks()
            self.assertTrue(len(chunks) >= 1)
            chunk = next(iter(chunks.values()))
            self.assertEqual(chunk.file_path, "src/sub module/my worker.py")
        finally:
            temp_dir.cleanup()

    def test_atomic_cache_write_protection_against_interruption(self):
        # Verify that an interrupted cache write leaves the prior valid cache untouched
        temp_dir = tempfile.TemporaryDirectory()
        repo_path = Path(temp_dir.name)
        cache_dir = repo_path / ".cache"

        try:
            (repo_path / "src").mkdir(parents=True)
            (repo_path / "src" / "mod.py").write_text("def first(): pass\n", encoding="utf-8")

            indexer = IncrementalIndexer(repo_path, cache_dir=cache_dir)
            report1 = indexer.index()
            self.assertEqual(report1.total_files, 1)

            # Read initial manifest and chunk content
            initial_manifest_content = indexer.manifest_path.read_text(encoding="utf-8")
            initial_chunks_content = indexer.chunks_path.read_text(encoding="utf-8")

            # Simulate a crash/interruption during atomic write of chunks.json
            with patch("json.dump", side_effect=KeyboardInterrupt("Simulated write interruption")):
                with self.assertRaises(KeyboardInterrupt):
                    indexer.save_index(indexer.load_manifest(), indexer.load_chunks())

            # Prior cache files must be preserved and undamaged!
            self.assertEqual(indexer.manifest_path.read_text(encoding="utf-8"), initial_manifest_content)
            self.assertEqual(indexer.chunks_path.read_text(encoding="utf-8"), initial_chunks_content)

            # Temporary partial files must not be left behind
            tmp_files = list(cache_dir.glob(".*.tmp"))
            self.assertEqual(len(tmp_files), 0)
        finally:
            temp_dir.cleanup()

    def test_indexing_failures_are_visible_in_report_and_cli(self):
        temp_dir = tempfile.TemporaryDirectory()
        repo_path = Path(temp_dir.name)
        cache_dir = repo_path / ".cache"

        try:
            (repo_path / "src").mkdir(parents=True)
            (repo_path / "src" / "valid.py").write_text("def ok(): pass\n", encoding="utf-8")
            (repo_path / "src" / "crash.py").write_text("def will_crash(): pass\n", encoding="utf-8")

            indexer = IncrementalIndexer(repo_path, cache_dir=cache_dir)

            # Force chunker to crash only on crash.py
            original_chunk_file = indexer.chunker.chunk_file

            def crashing_chunker(content, rel_file_path, language, source_revision=None):
                if "crash.py" in rel_file_path:
                    raise ValueError("Simulated unrecoverable token error")
                return original_chunk_file(content, rel_file_path, language, source_revision)

            indexer.chunker.chunk_file = crashing_chunker

            report = indexer.index()

            # Error must be visibly captured in failed_files
            self.assertIn("src/crash.py", report.failed_files)
            self.assertIn("ValueError", report.failed_files["src/crash.py"])
            self.assertFalse(report.is_fresh())

            # Valid file must succeed
            self.assertIn("src/valid.py", report.added_files)

            # Crash file must not register chunks or leave stale data
            chunks = indexer.load_chunks()
            crash_chunks = [c for c in chunks.values() if c.file_path == "src/crash.py"]
            self.assertEqual(len(crash_chunks), 0)
        finally:
            temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
