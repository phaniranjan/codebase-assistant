"""Unit tests for code and test proposal CLI commands."""

import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

from click.testing import CliRunner

from codebase_assistant.cli import main
from codebase_assistant.ingestion.indexer import IncrementalIndexer


class TestGenerationCLI(unittest.TestCase):
    """Test suite for propose-code and propose-tests CLI commands."""

    def setUp(self):
        self.runner = CliRunner()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.tmp_dir.name)

        # Create sample files
        src_dir = self.repo_dir / "src"
        src_dir.mkdir(parents=True)
        (src_dir / "auth.py").write_text(
            "def authenticate(token: str) -> bool:\n    return token == 'valid'\n",
            encoding="utf-8",
        )
        (src_dir / "calculator.py").write_text(
            "def add(a: int, b: int) -> int:\n    return a + b\n",
            encoding="utf-8",
        )

        # Index repository
        indexer = IncrementalIndexer(self.repo_dir)
        indexer.index()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_propose_code_cli_mock_success(self):
        res = self.runner.invoke(
            main,
            [
                "propose-code",
                "Add token validation function",
                str(self.repo_dir),
                "--provider",
                "mock",
                "--file",
                "src/auth.py",
            ],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Proposal #1", res.output)
        self.assertIn("src/auth.py", res.output)
        self.assertIn("AST Syntax: Valid Python", res.output)
        self.assertIn("--- Proposed Unified Diff ---", res.output)

    def test_propose_code_cli_output_patch(self):
        external_dir = Path(tempfile.mkdtemp())
        patch_file = external_dir / "auth_feature.patch"
        try:
            res = self.runner.invoke(
                main,
                [
                    "propose-code",
                    "Add logout method",
                    "--repo-path",
                    str(self.repo_dir),
                    "--file",
                    "src/auth.py",
                    "--provider",
                    "mock",
                    "--output-patch",
                    str(patch_file),
                ],
            )
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertIn("Saved unified diff patch to:", res.output)
            self.assertTrue(patch_file.is_file())
            content = patch_file.read_text(encoding="utf-8")
            self.assertIn("--- a/src/auth.py", content)
            self.assertIn("+++ b/src/auth.py", content)
        finally:
            shutil.rmtree(external_dir, ignore_errors=True)

    def test_propose_code_cli_rejects_internal_patch_path(self):
        internal_patch = self.repo_dir / "leak.patch"
        res = self.runner.invoke(
            main,
            [
                "propose-code",
                "Add logout method",
                str(self.repo_dir),
                "--file",
                "src/auth.py",
                "--provider",
                "mock",
                "--output-patch",
                str(internal_patch),
            ],
        )
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Read-only invariant violation", res.output)
        self.assertFalse(internal_patch.exists())

    def test_propose_tests_cli_mock_success(self):
        res = self.runner.invoke(
            main,
            [
                "propose-tests",
                "src/calculator.py",
                str(self.repo_dir),
                "--provider",
                "mock",
            ],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Proposal #1", res.output)
        self.assertIn("test_calculator.py", res.output)
        self.assertIn("AST Syntax: Valid Python", res.output)
        self.assertIn("--- Proposed Unified Diff ---", res.output)

    def test_propose_tests_cli_framework_override(self):
        res = self.runner.invoke(
            main,
            [
                "propose-tests",
                "src/calculator.py",
                "--repo-path",
                str(self.repo_dir),
                "--framework",
                "pytest",
                "--provider",
                "mock",
            ],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("test_calculator.py", res.output)
        self.assertIn("AST Syntax: Valid Python", res.output)

    def test_propose_tests_cli_rejects_internal_patch_path(self):
        internal_patch = self.repo_dir / "tests.patch"
        res = self.runner.invoke(
            main,
            [
                "propose-tests",
                "src/calculator.py",
                str(self.repo_dir),
                "--provider",
                "mock",
                "--output-patch",
                str(internal_patch),
            ],
        )
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Read-only invariant violation", res.output)
        self.assertFalse(internal_patch.exists())

    def test_propose_cli_rejects_internal_cache_dir(self):
        internal_cache = self.repo_dir / "custom_cache"
        res_code = self.runner.invoke(
            main,
            [
                "propose-code",
                "Add helper",
                str(self.repo_dir),
                "--file",
                "src/auth.py",
                "--provider",
                "mock",
                "--cache-dir",
                str(internal_cache),
            ],
        )
        self.assertEqual(res_code.exit_code, 1)
        self.assertIn("Read-only invariant violation", res_code.output)
        self.assertIn("is located inside the target repository", res_code.output)
        self.assertFalse(internal_cache.exists())

        res_tests = self.runner.invoke(
            main,
            [
                "propose-tests",
                "src/calculator.py",
                str(self.repo_dir),
                "--provider",
                "mock",
                "--cache-dir",
                str(internal_cache),
            ],
        )
        self.assertEqual(res_tests.exit_code, 1)
        self.assertIn("Read-only invariant violation", res_tests.output)
        self.assertIn("is located inside the target repository", res_tests.output)
        self.assertFalse(internal_cache.exists())

    def test_propose_preserves_target_repo_with_indexed_chunks_and_empty_vector_store(self):
        # 1. Target repository has indexed chunks but NO vector store
        chunks_file = self.repo_dir / ".codebase_assistant" / "chunks.json"
        self.assertTrue(chunks_file.is_file(), "Expected indexed chunks in test fixture")

        vector_file = self.repo_dir / ".codebase_assistant" / "vector_index.json"
        if vector_file.exists():
            vector_file.unlink()

        # 2. Snapshot target repository file tree with cryptographic SHA-256 hashes
        def snapshot_tree(root: Path) -> dict[str, str]:
            hashes = {}
            for p in root.rglob("*"):
                if p.is_file():
                    hashes[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
            return hashes

        pre_snapshot = snapshot_tree(self.repo_dir)

        # 3. Run propose-code (triggers manager.sync() when vector store is empty)
        res_code = self.runner.invoke(
            main,
            [
                "propose-code",
                "Implement secure password hashing",
                str(self.repo_dir),
                "--file",
                "src/auth.py",
                "--provider",
                "mock",
            ],
        )
        self.assertEqual(res_code.exit_code, 0, res_code.output)
        self.assertIn("Proposal #1", res_code.output)

        post_code_snapshot = snapshot_tree(self.repo_dir)
        self.assertEqual(
            pre_snapshot,
            post_code_snapshot,
            "Target repository files or hashes changed after propose-code!",
        )

        # 4. Run propose-tests
        res_test = self.runner.invoke(
            main,
            [
                "propose-tests",
                "src/calculator.py",
                str(self.repo_dir),
                "--provider",
                "mock",
            ],
        )
        self.assertEqual(res_test.exit_code, 0, res_test.output)
        self.assertIn("Proposal #1", res_test.output)

        post_test_snapshot = snapshot_tree(self.repo_dir)
        self.assertEqual(
            pre_snapshot,
            post_test_snapshot,
            "Target repository files or hashes changed after propose-tests!",
        )


if __name__ == "__main__":
    unittest.main()
