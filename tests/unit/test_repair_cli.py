"""Unit tests for the CLI repair command."""

import json
from pathlib import Path
import tempfile
import unittest
from click.testing import CliRunner

from codebase_assistant.cli import main
from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.ingestion.indexer import IncrementalIndexer


class TestRepairCLI(unittest.TestCase):
    """Test suite verifying the CLI repair command and options."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.tmp_dir.name)

        src_dir = self.repo_dir / "src"
        src_dir.mkdir(parents=True)
        self.calc_file = src_dir / "calc.py"
        self.calc_file.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")

        tests_dir = self.repo_dir / "tests" / "unit"
        tests_dir.mkdir(parents=True)
        self.test_calc_file = tests_dir / "test_calc.py"
        self.test_calc_file.write_text(
            "import unittest\nfrom src.calc import add\n\n"
            "class TestCalc(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 1), 2)\n",
            encoding="utf-8",
        )

        self.cache_dir = Path(tempfile.mkdtemp())
        indexer = IncrementalIndexer(self.repo_dir, cache_dir=self.cache_dir)
        indexer.index()

        self.runner = CliRunner()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_cli_repair_success(self):
        res = self.runner.invoke(
            main,
            [
                "repair",
                "--repo", str(self.repo_dir),
                "--task", "Add multiply function in src/calc.py",
                "--test-command", "python -m unittest tests/unit/test_calc.py",
                "--provider", "mock",
                "--auto-approve",
                "--cache-dir", str(self.cache_dir),
            ],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("SUCCESS", res.output)
        self.assertIn("Final Verified Patch", res.output)

    def test_cli_repair_json_output(self):
        res = self.runner.invoke(
            main,
            [
                "repair",
                "--repo", str(self.repo_dir),
                "--task", "Add helper function in src/calc.py",
                "--test-command", "python -m unittest tests/unit/test_calc.py",
                "--provider", "mock",
                "--auto-approve",
                "--cache-dir", str(self.cache_dir),
                "--json",
            ],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        data = json.loads(res.output)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["task"], "Add helper function in src/calc.py")
        self.assertIn("attempts", data)

    def test_cli_repair_report_file(self):
        report_path = self.repo_dir / "repair_report.md"
        res = self.runner.invoke(
            main,
            [
                "repair",
                "--repo", str(self.repo_dir),
                "--task", "Add helper function in src/calc.py",
                "--test-command", "python -m unittest tests/unit/test_calc.py",
                "--provider", "mock",
                "--auto-approve",
                "--cache-dir", str(self.cache_dir),
                "--report-file", str(report_path),
            ],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertTrue(report_path.exists())
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("# Agentic Repair Workflow Report:", content)
        self.assertIn("🟢 SUCCESS", content)


if __name__ == "__main__":
    unittest.main()
