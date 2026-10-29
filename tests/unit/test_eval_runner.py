"""Unit tests for EvaluationRunner and EvaluationReporter."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codebase_assistant.evaluation.dataset import EvalCategory, GoldenDataset, GoldenTestCase
from codebase_assistant.evaluation.report import EvaluationReporter
from codebase_assistant.evaluation.runner import EvaluationRunner


class TestEvaluationRunner(unittest.TestCase):
    """Test suite for EvaluationRunner and report generation."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.tmp_dir.name)

        # Create sample codebase
        src_dir = self.repo_dir / "src"
        src_dir.mkdir(parents=True)
        (src_dir / "auth.py").write_text("def authenticate(): return True\n", encoding="utf-8")
        (src_dir / "db.py").write_text("class Database: pass\n", encoding="utf-8")

        # Create sample dataset
        self.case1 = GoldenTestCase(
            id="test-001",
            question="Where is authenticate defined?",
            category=EvalCategory.SYMBOL_LOOKUP,
            is_answerable=True,
            expected_files=["src/auth.py"],
            expected_symbols=["authenticate"],
            expected_facts=["authenticate returns True"],
        )
        self.case2 = GoldenTestCase(
            id="test-002",
            question="How is Kubernetes cluster ingress configured?",
            category=EvalCategory.UNANSWERABLE,
            is_answerable=False,
            expected_files=[],
            expected_facts=["No kubernetes"],
        )
        self.dataset = GoldenDataset(
            version="1.0.0",
            target_repository="test-repo",
            cases=[self.case1, self.case2],
        )

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_offline_evaluation_runner_execution(self):
        runner = EvaluationRunner(
            repository_path=self.repo_dir,
            is_live=False,
        )
        report = runner.run(dataset=self.dataset, mode="hybrid", top_k=2)

        self.assertFalse(report.is_live)
        self.assertIn("Mock", report.provider_name)
        self.assertEqual(report.total_cases, 2)
        self.assertEqual(report.evaluated_cases, 2)
        self.assertIn("file_recall_at_k", report.macro_metrics)
        self.assertIn("abstention_recall", report.macro_metrics)
        self.assertEqual(len(report.case_results), 2)

        # Verify Report serialization
        json_str = EvaluationReporter.to_json(report)
        self.assertIn('"run_id"', json_str)
        self.assertIn('"quality_gate"', json_str)

        md_str = EvaluationReporter.to_markdown(report)
        self.assertIn("# RAG Evaluation Summary Report", md_str)
        self.assertIn("Quality Gates & Policy Enforcement", md_str)
        self.assertIn("Macro Aggregate Metrics", md_str)

    def test_live_evaluation_fails_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                EvaluationRunner(repository_path=self.repo_dir, is_live=True)
            self.assertIn("GEMINI_API_KEY environment variable is not set", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
