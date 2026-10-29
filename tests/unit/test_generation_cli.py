"""Unit tests for code and test proposal CLI commands."""

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


if __name__ == "__main__":
    unittest.main()
