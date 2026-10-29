"""Integration test verifying end-to-end RAG evaluation and read-only protection of ai-evaluation."""

import hashlib
from pathlib import Path
import tempfile
import unittest

from codebase_assistant.evaluation.dataset import GoldenDataset
from codebase_assistant.evaluation.report import EvaluationReporter
from codebase_assistant.evaluation.runner import EvaluationRunner
from codebase_assistant.ingestion.scanner import RepositoryScanner


class TestAiEvalFrameworkIntegration(unittest.TestCase):
    """End-to-end evaluation suite over the read-only ai-evaluation repository."""

    @classmethod
    def setUpClass(cls):
        candidates = [
            Path("../ai_evaluation").resolve(),
            Path("../../ai_evaluation").resolve(),
            Path("/home/phaniranjan/AI/ai_evaluation").resolve(),
            Path(r"\\wsl.localhost\Ubuntu-26.04\home\phaniranjan\AI\ai_evaluation").resolve(),
        ]
        cls.ai_eval_dir = None
        for cand in candidates:
            if cand.is_dir() and (cand / "src" / "ai_evaluation").is_dir():
                cls.ai_eval_dir = cand
                break

    def setUp(self):
        if self.ai_eval_dir is None:
            self.skipTest("ai-evaluation repository not found on local path.")

        # 1. Take cryptographic SHA-256 snapshot of all target repository files
        scanner = RepositoryScanner(self.ai_eval_dir)
        self.eligible_files, _, _ = scanner.scan()

        self.initial_hashes = {}
        for rel_p in self.eligible_files:
            abs_p = self.ai_eval_dir / rel_p
            if abs_p.is_file():
                content = abs_p.read_bytes()
                self.initial_hashes[rel_p.as_posix()] = hashlib.sha256(content).hexdigest()

        # 2. Allocate isolated temporary cache directory outside target repository
        self.temp_cache = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.temp_cache.name)

    def tearDown(self):
        self.temp_cache.cleanup()

        # Strict cryptographic verification that target repository was 100% read-only
        current_hashes = {}
        for rel_p in self.eligible_files:
            abs_p = self.ai_eval_dir / rel_p
            if abs_p.is_file():
                content = abs_p.read_bytes()
                current_hashes[rel_p.as_posix()] = hashlib.sha256(content).hexdigest()

        self.assertEqual(
            self.initial_hashes,
            current_hashes,
            "Target repository was modified during evaluation! SHA-256 mismatch detected.",
        )

    def test_offline_golden_dataset_evaluation_and_hard_gates(self):
        dataset_path = Path("data/golden_dataset.json")
        self.assertTrue(dataset_path.is_file(), "Golden dataset missing at data/golden_dataset.json")
        dataset = GoldenDataset.load_from_file(dataset_path)

        runner = EvaluationRunner(
            repository_path=self.ai_eval_dir,
            cache_dir=self.cache_dir,
            is_live=False,
        )

        # 1. Evaluate Hybrid mode
        report = runner.run(dataset=dataset, mode="hybrid", top_k=5)

        self.assertEqual(report.total_cases, 25)
        self.assertEqual(report.evaluated_cases, 25)
        self.assertFalse(report.is_live)

        # Hard Gate verification: 100% citation validity, boundaries, abstention recall, and injection containment
        self.assertTrue(
            report.quality_gate.hard_gates_passed(),
            f"Hard gates failed in evaluation:\n{report.quality_gate.failing_cases}",
        )

        # Verify macro metrics
        macro = report.macro_metrics
        self.assertEqual(macro["citation_validity_rate"], 1.0)
        self.assertEqual(macro["citation_boundary_rate"], 1.0)
        self.assertEqual(macro["abstention_recall"], 1.0)
        self.assertGreater(macro["file_hit_rate"], 0.70)

        # Verify Report serialization
        report_json = EvaluationReporter.to_json(report)
        self.assertIn('"target_repository": "ai-evaluation"', report_json)

        report_md = EvaluationReporter.to_markdown(report)
        self.assertIn("# RAG Evaluation Summary Report", report_md)
        self.assertIn("Quality Gates & Policy Enforcement", report_md)


if __name__ == "__main__":
    unittest.main()
