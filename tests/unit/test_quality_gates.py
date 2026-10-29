"""Unit tests for QualityGatePolicy and quality gate thresholds."""

import unittest

from codebase_assistant.evaluation.dataset import EvalCategory
from codebase_assistant.evaluation.metrics import (
    AnswerabilityMetrics,
    CaseEvaluationResult,
    CitationMetrics,
    PromptInjectionMetrics,
    RetrievalMetrics,
)
from codebase_assistant.evaluation.quality_gates import (
    QualityGatePolicy,
    QualityGateThresholds,
)


class TestQualityGates(unittest.TestCase):
    """Test suite for hard gates, provisional thresholds, and failure reporting."""

    def _create_mock_result(
        self,
        case_id: str,
        is_answerable: bool,
        file_recall: float = 1.0,
        citation_validity: float = 1.0,
        citation_boundary: float = 1.0,
        classification: str = "TP",
        is_contained: bool = True,
    ) -> CaseEvaluationResult:
        retrieval = RetrievalMetrics(
            file_recall=file_recall if is_answerable else None,
            file_precision=1.0,
            file_mrr=1.0,
            file_hit_rate=1.0,
            chunk_precision=1.0,
            chunk_hit_rate=1.0,
        )
        citations = CitationMetrics(
            validity_rate=citation_validity,
            boundary_rate=citation_boundary,
            coverage=1.0,
            total_citations=1,
            valid_citations=1 if citation_validity == 1.0 else 0,
        )
        answerability = AnswerabilityMetrics(
            classification=classification,
            is_abstention=(classification in ("TN", "FN")),
            is_correct_abstention=(classification == "TN"),
            is_unnecessary_abstention=(classification == "FN"),
            is_missed_abstention=(classification == "FP"),
        )
        injection = PromptInjectionMetrics(is_contained=is_contained)

        return CaseEvaluationResult(
            case_id=case_id,
            category=EvalCategory.ARCHITECTURE if is_answerable else EvalCategory.UNANSWERABLE,
            is_answerable=is_answerable,
            retrieval=retrieval,
            citations=citations,
            answerability=answerability,
            injection=injection,
        )

    def test_all_passing_results_pass_quality_gate(self):
        r1 = self._create_mock_result("c1", is_answerable=True, classification="TP")
        r2 = self._create_mock_result("c2", is_answerable=False, classification="TN")

        policy = QualityGatePolicy()
        res = policy.evaluate([r1, r2])

        self.assertTrue(res.passed)
        self.assertTrue(res.hard_gates_passed())
        self.assertEqual(len(res.failing_cases), 0)

    def test_hard_gate_citation_validity_failure(self):
        # Result has citation validity < 1.0
        r1 = self._create_mock_result("c1", is_answerable=True, citation_validity=0.5)

        policy = QualityGatePolicy()
        res = policy.evaluate([r1])

        self.assertFalse(res.passed)
        self.assertFalse(res.hard_gates_passed())
        self.assertEqual(len(res.failing_cases), 1)
        self.assertIn("Citation validity", res.failing_cases[0]["failures"][0])

    def test_hard_gate_prompt_injection_failure(self):
        r1 = self._create_mock_result("c1", is_answerable=True, is_contained=False)

        policy = QualityGatePolicy()
        res = policy.evaluate([r1])

        self.assertFalse(res.passed)
        self.assertFalse(res.hard_gates_passed())
        self.assertEqual(len(res.failing_cases), 1)
        self.assertIn("Prompt injection breached", res.failing_cases[0]["failures"][0])

    def test_hard_gate_missed_abstention_failure(self):
        # Unanswerable question answered (FP)
        r_fp = self._create_mock_result("c_unans", is_answerable=False, classification="FP")

        policy = QualityGatePolicy()
        res = policy.evaluate([r_fp])

        self.assertFalse(res.passed)
        self.assertFalse(res.hard_gates_passed())
        self.assertEqual(len(res.failing_cases), 1)
        self.assertIn("Missed abstention", res.failing_cases[0]["failures"][0])

    def test_configurable_retrieval_threshold_failure(self):
        # Low recall (0.5 < 0.80) but 100% hard gates passed
        r1 = self._create_mock_result("c1", is_answerable=True, file_recall=0.5)

        policy = QualityGatePolicy(QualityGateThresholds(min_file_recall_at_k=0.80))
        res = policy.evaluate([r1])

        self.assertFalse(res.passed)  # Overall fails due to provisional retrieval gate
        self.assertTrue(res.hard_gates_passed())  # Hard gates still passed!


if __name__ == "__main__":
    unittest.main()
