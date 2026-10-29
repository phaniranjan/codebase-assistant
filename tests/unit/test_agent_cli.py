"""Unit tests for the agent CLI command."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from click.testing import CliRunner

from codebase_assistant.cli import main
from codebase_assistant.ingestion.indexer import IncrementalIndexer


class TestAgentCLI(unittest.TestCase):
    """Test suite for 'codebase-assistant agent' command-line interface."""

    def setUp(self):
        self.runner = CliRunner()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.tmp_dir.name)

        src_dir = self.repo_dir / "src"
        src_dir.mkdir(parents=True)
        (src_dir / "calculator.py").write_text(
            "def multiply(a: int, b: int) -> int:\n    return a * b\n",
            encoding="utf-8",
        )

        indexer = IncrementalIndexer(self.repo_dir)
        indexer.index()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_agent_cli_mock_execution(self):
        res = self.runner.invoke(
            main,
            [
                "agent",
                "Find the multiply function in the repository",
                str(self.repo_dir),
                "--provider",
                "mock",
                "--auto-approve",
            ],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Executing Agent Loop", res.output)
        self.assertIn("[Step 1]", res.output)
        self.assertIn("Action:", res.output)
        self.assertIn("Stopping Reason: COMPLETED", res.output)
        self.assertIn("--- Final Answer ---", res.output)

    def test_agent_cli_rejects_internal_cache_dir(self):
        internal_cache = self.repo_dir / "agent_cache"
        res = self.runner.invoke(
            main,
            [
                "agent",
                "Inspect repository",
                str(self.repo_dir),
                "--provider",
                "mock",
                "--cache-dir",
                str(internal_cache),
            ],
        )
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Read-only invariant violation", res.output)
        self.assertFalse(internal_cache.exists())

    def test_agent_cli_preserves_target_repo_read_only(self):
        # Cryptographic snapshot of all files in target repository
        def snapshot_tree(root: Path) -> dict[str, str]:
            hashes = {}
            for p in root.rglob("*"):
                if p.is_file():
                    hashes[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
            return hashes

        pre_snapshot = snapshot_tree(self.repo_dir)

        res = self.runner.invoke(
            main,
            [
                "agent",
                "Locate calculator symbols",
                str(self.repo_dir),
                "--provider",
                "mock",
                "--auto-approve",
            ],
        )
        self.assertEqual(res.exit_code, 0, res.output)

        post_snapshot = snapshot_tree(self.repo_dir)
        self.assertEqual(
            pre_snapshot,
            post_snapshot,
            "Target repository was modified during read-only agent execution!",
        )


if __name__ == "__main__":
    unittest.main()
