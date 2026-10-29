"""Unit tests for Codebase Assistant CLI commands."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from codebase_assistant.cli import main
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.ingestion.indexer import IncrementalIndexer


class TestCLI(unittest.TestCase):
    """Test suite for CLI arguments, fallback policies, and streaming."""

    def setUp(self):
        self.runner = CliRunner()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.tmp_dir.name)

        # Create a sample python file in repo
        src_dir = self.repo_dir / "src"
        src_dir.mkdir(parents=True)
        sample_file = src_dir / "auth.py"
        sample_file.write_text(
            "def authenticate_user(token: str):\n    return token == 'valid'\n",
            encoding="utf-8",
        )

        # Pre-index the sample repository
        indexer = IncrementalIndexer(self.repo_dir)
        indexer.index()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_repo_path_positional_and_option_in_search(self):
        # 1. Positional repository path
        res_pos = self.runner.invoke(
            main,
            ["search", "authenticate", str(self.repo_dir), "--provider", "mock"],
        )
        self.assertEqual(res_pos.exit_code, 0, res_pos.output)
        self.assertIn("auth.py", res_pos.output)

        # 2. Option --repo-path
        res_opt = self.runner.invoke(
            main,
            ["search", "authenticate", "--repo-path", str(self.repo_dir), "--provider", "mock"],
        )
        self.assertEqual(res_opt.exit_code, 0, res_opt.output)
        self.assertIn("auth.py", res_opt.output)

    def test_repo_path_positional_and_option_in_ask(self):
        # 1. Positional repository path
        res_pos = self.runner.invoke(
            main,
            ["ask", "How does authenticate work?", str(self.repo_dir), "--provider", "mock"],
        )
        self.assertEqual(res_pos.exit_code, 0, res_pos.output)
        self.assertIn("auth.py", res_pos.output)

        # 2. Option --repo-path
        res_opt = self.runner.invoke(
            main,
            ["ask", "How does authenticate work?", "--repo-path", str(self.repo_dir), "--provider", "mock"],
        )
        self.assertEqual(res_opt.exit_code, 0, res_opt.output)
        self.assertIn("auth.py", res_opt.output)

    def test_ask_stream_mode(self):
        res = self.runner.invoke(
            main,
            ["ask", "How does authenticate work?", str(self.repo_dir), "--provider", "mock", "--stream"],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Answer:", res.output)
        self.assertIn("auth.py", res.output)
        self.assertIn("Source Citations", res.output)

    def test_gemini_fails_without_fallback_when_key_missing(self):
        # Ensure GEMINI_API_KEY is not set
        with patch.dict(os.environ, {}, clear=True):
            # 1. embed command
            res_embed = self.runner.invoke(
                main,
                ["embed", str(self.repo_dir), "--provider", "gemini"],
            )
            self.assertEqual(res_embed.exit_code, 1)
            self.assertIn("Failed to initialize Gemini embedding provider", res_embed.output)

            # 2. search command
            res_search = self.runner.invoke(
                main,
                ["search", "authenticate", str(self.repo_dir), "--provider", "gemini"],
            )
            self.assertEqual(res_search.exit_code, 1)
            self.assertIn("Failed to initialize Gemini embedding provider", res_search.output)

            # 3. ask command
            res_ask = self.runner.invoke(
                main,
                ["ask", "test question", str(self.repo_dir), "--provider", "gemini"],
            )
            self.assertEqual(res_ask.exit_code, 1)
            self.assertIn("Failed to initialize Gemini client/provider", res_ask.output)

    def test_gemini_falls_back_when_allow_fallback_flag_provided(self):
        # With --allow-fallback, commands should succeed with fallback notice
        with patch.dict(os.environ, {}, clear=True):
            # 1. embed command with --allow-fallback
            res_embed = self.runner.invoke(
                main,
                ["embed", str(self.repo_dir), "--provider", "gemini", "--allow-fallback"],
            )
            self.assertEqual(res_embed.exit_code, 0)
            self.assertIn("Falling back to MockEmbeddingProvider", res_embed.output)

            # 2. search command with --allow-fallback
            res_search = self.runner.invoke(
                main,
                ["search", "authenticate", str(self.repo_dir), "--provider", "gemini", "--allow-fallback"],
            )
            self.assertEqual(res_search.exit_code, 0)
            self.assertIn("Falling back to MockEmbeddingProvider", res_search.output)

            # 3. ask command with --allow-fallback
            res_ask = self.runner.invoke(
                main,
                ["ask", "test question", str(self.repo_dir), "--provider", "gemini", "--allow-fallback"],
            )
            self.assertEqual(res_ask.exit_code, 0)
            self.assertIn("Falling back to Mock providers", res_ask.output)

    def test_evaluate_cli_command_offline_and_reports(self):
        # Create a small sample dataset JSON
        sample_dataset = self.repo_dir / "test_dataset.json"
        sample_dataset.write_text("""{
  "version": "1.0.0",
  "target_repository": "test-repo",
  "cases": [
    {
      "id": "c1",
      "question": "How does auth work?",
      "category": "symbol_lookup",
      "is_answerable": true,
      "expected_files": ["src/auth.py"],
      "expected_symbols": ["authenticate_user"]
    }
  ]
}""", encoding="utf-8")

        external_out_dir = Path(tempfile.mkdtemp())
        json_out = external_out_dir / "report.json"
        md_out = external_out_dir / "report.md"

        try:
            res = self.runner.invoke(
                main,
                [
                    "evaluate",
                    str(self.repo_dir),
                    "--dataset", str(sample_dataset),
                    "--mode", "hybrid",
                    "--top-k", "2",
                    "--output-json", str(json_out),
                    "--output-markdown", str(md_out),
                ],
            )
            self.assertEqual(res.exit_code, 0, res.output)
            self.assertIn("Quality Gate Status", res.output)
            self.assertTrue(json_out.is_file())
            self.assertTrue(md_out.is_file())
        finally:
            import shutil
            shutil.rmtree(external_out_dir, ignore_errors=True)

    def test_evaluate_cli_rejects_internal_cache_dir(self):
        sample_dataset = self.repo_dir / "test_dataset.json"
        sample_dataset.write_text('{"version": "1.0.0", "cases": []}', encoding="utf-8")
        internal_cache = self.repo_dir / "my_internal_cache"

        res = self.runner.invoke(
            main,
            [
                "evaluate",
                str(self.repo_dir),
                "--dataset", str(sample_dataset),
                "--cache-dir", str(internal_cache),
            ],
        )
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Read-only invariant violation", res.output)

    def test_evaluate_cli_live_fails_without_key(self):
        sample_dataset = self.repo_dir / "test_dataset.json"
        sample_dataset.write_text("{\"cases\": []}", encoding="utf-8")

        with patch.dict(os.environ, {}, clear=True):
            res = self.runner.invoke(
                main,
                ["evaluate", str(self.repo_dir), "--dataset", str(sample_dataset), "--live"],
            )
            self.assertEqual(res.exit_code, 1)
            self.assertIn("GEMINI_API_KEY environment variable is not set", res.output)


if __name__ == "__main__":
    unittest.main()
