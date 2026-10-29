"""RAG evaluation framework and quality gate enforcement."""

from codebase_assistant.evaluation.dataset import (
    EvalCategory,
    GoldenDataset,
    GoldenTestCase,
    normalize_repo_path,
)
from codebase_assistant.evaluation.metrics import (
    AnswerabilityMetrics,
    CaseEvaluationResult,
    CitationMetrics,
    MetricsCalculator,
    PromptInjectionMetrics,
    RetrievalMetrics,
)
from codebase_assistant.evaluation.quality_gates import (
    GateCheckDetail,
    QualityGatePolicy,
    QualityGateResult,
    QualityGateThresholds,
)
from codebase_assistant.evaluation.report import EvaluationReporter
from codebase_assistant.evaluation.runner import (
    CategorySummary,
    EvaluationRunReport,
    EvaluationRunner,
)

__all__ = [
    "EvalCategory",
    "GoldenDataset",
    "GoldenTestCase",
    "normalize_repo_path",
    "RetrievalMetrics",
    "CitationMetrics",
    "AnswerabilityMetrics",
    "PromptInjectionMetrics",
    "CaseEvaluationResult",
    "MetricsCalculator",
    "QualityGateThresholds",
    "GateCheckDetail",
    "QualityGateResult",
    "QualityGatePolicy",
    "CategorySummary",
    "EvaluationRunReport",
    "EvaluationRunner",
    "EvaluationReporter",
]
