"""Evaluation runner orchestrating retrieval and grounded Q&A across the golden dataset."""

from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from codebase_assistant.config import get_settings
from codebase_assistant.embeddings.gemini import GeminiEmbeddingProvider
from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.evaluation.dataset import EvalCategory, GoldenDataset, GoldenTestCase
from codebase_assistant.evaluation.metrics import (
    CaseEvaluationResult,
    MetricsCalculator,
    RetrievalMetrics,
)
from codebase_assistant.evaluation.quality_gates import QualityGatePolicy, QualityGateResult, QualityGateThresholds
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.llm.client import GeminiClient
from codebase_assistant.llm.mock_client import MockGeminiClient
from codebase_assistant.llm.qa import CodebaseQAEngine
from codebase_assistant.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class CategorySummary(BaseModel):
    """Aggregate metric breakdown per evaluation category."""

    category: str
    total_cases: int
    evaluated_cases: int
    avg_file_recall: Optional[float] = None
    avg_file_mrr: float = 0.0
    avg_file_hit_rate: float = 0.0
    citation_validity_rate: float = 0.0
    abstention_accuracy: float = 0.0
    fact_coverage_score: float = 0.0


class EvaluationRunReport(BaseModel):
    """Top-level evaluation execution artifact."""

    run_id: str
    timestamp: str
    target_repository: str
    dataset_version: str
    retrieval_mode: str
    top_k: int
    is_live: bool
    provider_name: str
    model_name: str
    total_cases: int
    evaluated_cases: int
    total_duration_seconds: float
    total_prompt_tokens: int = 0
    macro_metrics: Dict[str, Any] = Field(default_factory=dict)
    category_summaries: Dict[str, CategorySummary] = Field(default_factory=dict)
    quality_gate: QualityGateResult
    case_results: List[CaseEvaluationResult] = Field(default_factory=list)


class EvaluationRunner:
    """Coordinates indexing, retrieval, QA engine execution, and metric calculation."""

    def __init__(
        self,
        repository_path: Path | str,
        cache_dir: Optional[Path | str] = None,
        is_live: bool = False,
        quality_thresholds: Optional[QualityGateThresholds] = None,
    ) -> None:
        self.repository_path = Path(repository_path).resolve()
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else self.repository_path / get_settings().cache_dir
        self.is_live = is_live
        self.policy = QualityGatePolicy(quality_thresholds)

        # 1. Initialize Providers strictly according to mode
        if self.is_live:
            settings = get_settings()
            if not settings.gemini_api_key:
                raise ValueError(
                    "Live evaluation explicitly requested (--live), but GEMINI_API_KEY environment variable is not set. "
                    "Cannot execute live evaluation without valid API key."
                )
            self.emb_provider = GeminiEmbeddingProvider()
            self.llm_client = GeminiClient()
            # Verify client initialization
            self.emb_provider._get_client()
            self.llm_client._get_client()
            self.provider_name = "Gemini (Official Google GenAI SDK)"
            self.model_name = self.llm_client.model_name
        else:
            self.emb_provider = MockEmbeddingProvider(dimension=256)
            self.llm_client = MockGeminiClient(model_name="mock-gemini-v1")
            self.provider_name = "Mock (Deterministic Offline Provider)"
            self.model_name = "mock-gemini-v1"

    def run(
        self,
        dataset: GoldenDataset,
        mode: Literal["hybrid", "dense_only", "sparse_only"] = "hybrid",
        top_k: int = 5,
        budget: int = 8192,
        rebuild_index: bool = False,
    ) -> EvaluationRunReport:
        """Execute evaluation over all cases in dataset."""
        start_time = time.time()
        run_id = f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

        # 1. Load or sync index & vector store
        indexer = IncrementalIndexer(self.repository_path, cache_dir=self.cache_dir)
        if rebuild_index:
            indexer.index(rebuild=True)
        chunks = indexer.load_chunks()
        if not chunks:
            indexer.index()
            chunks = indexer.load_chunks()

        manager = EmbeddingManager(self.repository_path, cache_dir=self.cache_dir, provider=self.emb_provider)
        vector_store = manager.load_vector_store()
        if mode in ("hybrid", "dense_only") and vector_store.count() == 0:
            manager.sync(rebuild=rebuild_index)
            vector_store = manager.load_vector_store()

        retriever = HybridRetriever(
            chunks=chunks,
            vector_store=vector_store,
            embedding_provider=self.emb_provider,
        )

        qa_engine = CodebaseQAEngine(
            retriever=retriever,
            llm_client=self.llm_client,
            default_budget=budget,
        )

        case_results: List[CaseEvaluationResult] = []
        total_tokens = 0

        # 2. Iterate and evaluate each golden case
        for case in dataset.cases:
            try:
                # Retrieval step
                retrieved_chunks = retriever.search(query=case.question, top_k=top_k, mode=mode)
                ret_metrics = MetricsCalculator.evaluate_retrieval(
                    retrieved_chunks=retrieved_chunks,
                    case=case,
                    top_k=top_k,
                )

                # Grounded QA step
                qa_resp = qa_engine.ask(
                    query=case.question,
                    top_k=top_k,
                    mode=mode,
                    budget=budget,
                )
                total_tokens += qa_resp.context_tokens

                # Abstention evaluation
                ans_metrics = MetricsCalculator.evaluate_abstention(qa_resp.answer, case)

                # Citation evaluation (accounting for abstention)
                cit_metrics = MetricsCalculator.evaluate_citations(
                    citations=qa_resp.citations,
                    assembled_chunks=retrieved_chunks,
                    case=case,
                    is_abstained=ans_metrics.is_abstention,
                )

                # Prompt injection evaluation
                inj_metrics = MetricsCalculator.evaluate_prompt_injection(qa_resp.answer, case)

                # Expected facts smoke test
                fact_score = MetricsCalculator.evaluate_fact_coverage(qa_resp.answer, case)

                case_results.append(
                    CaseEvaluationResult(
                        case_id=case.id,
                        category=case.category,
                        is_answerable=case.is_answerable,
                        retrieval=ret_metrics,
                        citations=cit_metrics,
                        answerability=ans_metrics,
                        injection=inj_metrics,
                        fact_coverage_score=fact_score,
                        answer_text=qa_resp.answer,
                    )
                )
            except Exception as e:
                logger.error("Error evaluating case %s: %s", case.id, e, exc_info=True)
                case_results.append(
                    CaseEvaluationResult(
                        case_id=case.id,
                        category=case.category,
                        is_answerable=case.is_answerable,
                        retrieval=RetrievalMetrics(),
                        citations=MetricsCalculator.evaluate_citations([], [], case),
                        answerability=MetricsCalculator.evaluate_abstention("", case),
                        error=str(e),
                    )
                )

        # 3. Evaluate Quality Gates
        gate_result = self.policy.evaluate(case_results)

        # 4. Compute Category Summaries & Macro Metrics
        category_summaries = self._compute_category_summaries(case_results)
        macro_metrics = self._compute_macro_metrics(case_results, gate_result)

        duration = time.time() - start_time

        return EvaluationRunReport(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            target_repository=dataset.target_repository,
            dataset_version=dataset.version,
            retrieval_mode=mode,
            top_k=top_k,
            is_live=self.is_live,
            provider_name=self.provider_name,
            model_name=self.model_name,
            total_cases=len(dataset.cases),
            evaluated_cases=len(case_results) - len(gate_result.skipped_cases),
            total_duration_seconds=round(duration, 3),
            total_prompt_tokens=total_tokens,
            macro_metrics=macro_metrics,
            category_summaries=category_summaries,
            quality_gate=gate_result,
            case_results=case_results,
        )

    def _compute_category_summaries(
        self, results: List[CaseEvaluationResult]
    ) -> Dict[str, CategorySummary]:
        by_cat: Dict[str, List[CaseEvaluationResult]] = {}
        for r in results:
            by_cat.setdefault(r.category.value, []).append(r)

        summaries: Dict[str, CategorySummary] = {}
        for cat_name, cat_res in by_cat.items():
            valid_recalls = [r.retrieval.file_recall for r in cat_res if r.retrieval.file_recall is not None]
            avg_recall = sum(valid_recalls) / len(valid_recalls) if valid_recalls else None
            valid_mrrs = [r.retrieval.file_mrr for r in cat_res if r.retrieval.file_mrr is not None]
            avg_mrr = sum(valid_mrrs) / len(valid_mrrs) if valid_mrrs else 0.0
            valid_hits = [r.retrieval.file_hit_rate for r in cat_res if r.retrieval.file_hit_rate is not None]
            avg_hit = sum(valid_hits) / len(valid_hits) if valid_hits else 0.0
            cit_val = sum(r.citations.validity_rate for r in cat_res) / len(cat_res)
            fact_cov = sum(r.fact_coverage_score for r in cat_res) / len(cat_res)

            # Abstention accuracy within category
            correct_classifications = sum(
                1 for r in cat_res if r.answerability.classification in ("TP", "TN")
            )
            abstention_acc = correct_classifications / len(cat_res) if cat_res else 0.0

            summaries[cat_name] = CategorySummary(
                category=cat_name,
                total_cases=len(cat_res),
                evaluated_cases=sum(1 for r in cat_res if not r.error),
                avg_file_recall=round(avg_recall, 4) if avg_recall is not None else None,
                avg_file_mrr=round(avg_mrr, 4),
                avg_file_hit_rate=round(avg_hit, 4),
                citation_validity_rate=round(cit_val, 4),
                abstention_accuracy=round(abstention_acc, 4),
                fact_coverage_score=round(fact_cov, 4),
            )
        return summaries

    def _compute_macro_metrics(
        self, results: List[CaseEvaluationResult], gate: QualityGateResult
    ) -> Dict[str, Any]:
        valid_recalls = [r.retrieval.file_recall for r in results if r.retrieval.file_recall is not None]
        avg_recall = sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0.0
        valid_mrrs = [r.retrieval.file_mrr for r in results if r.retrieval.file_mrr is not None]
        avg_mrr = sum(valid_mrrs) / len(valid_mrrs) if valid_mrrs else 0.0
        valid_hits = [r.retrieval.file_hit_rate for r in results if r.retrieval.file_hit_rate is not None]
        avg_hit_rate = sum(valid_hits) / len(valid_hits) if valid_hits else 0.0

        valid_cits = [r.citations.validity_rate for r in results if r.is_answerable or r.citations.total_citations > 0]
        avg_cit_val = sum(valid_cits) / len(valid_cits) if valid_cits else 1.0

        valid_bounds = [r.citations.boundary_rate for r in results if r.citations.valid_citations > 0]
        avg_cit_bound = sum(valid_bounds) / len(valid_bounds) if valid_bounds else 1.0

        tp = sum(1 for r in results if r.answerability.classification == "TP")
        fp = sum(1 for r in results if r.answerability.classification == "FP")
        tn = sum(1 for r in results if r.answerability.classification == "TN")
        fn = sum(1 for r in results if r.answerability.classification == "FN")

        abstention_recall = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        unnecessary_abstention = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        answerability_acc = (tp + tn) / len(results) if results else 0.0

        return {
            "file_recall_at_k": round(avg_recall, 4),
            "file_mrr": round(avg_mrr, 4),
            "file_hit_rate": round(avg_hit_rate, 4),
            "citation_validity_rate": round(avg_cit_val, 4),
            "citation_boundary_rate": round(avg_cit_bound, 4),
            "abstention_recall": round(abstention_recall, 4),
            "unnecessary_abstention_rate": round(unnecessary_abstention, 4),
            "answerability_accuracy": round(answerability_acc, 4),
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
        }
