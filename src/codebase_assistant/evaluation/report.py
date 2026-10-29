"""Generators for machine-readable JSON and human-readable Markdown evaluation reports."""

import json
from pathlib import Path
from typing import Optional

from codebase_assistant.evaluation.runner import EvaluationRunReport


class EvaluationReporter:
    """Renders evaluation run artifacts into standardized JSON and Markdown formats."""

    @classmethod
    def to_json(cls, report: EvaluationRunReport) -> str:
        """Serialize evaluation report to deterministic JSON."""
        return json.dumps(report.model_dump(), indent=2, ensure_ascii=False)

    @classmethod
    def to_markdown(cls, report: EvaluationRunReport) -> str:
        """Render a concise, human-readable Markdown summary report."""
        status_badge = "**PASSED** ✅" if report.quality_gate.passed else "**FAILED** ❌"
        hard_badge = "**PASSED** ✅" if report.quality_gate.hard_gates_passed() else "**FAILED** ❌"

        lines = [
            f"# RAG Evaluation Summary Report — {report.run_id}",
            "",
            f"- **Target Repository**: `{report.target_repository}`",
            f"- **Dataset Version**: `{report.dataset_version}`",
            f"- **Timestamp**: `{report.timestamp}`",
            f"- **Retrieval Mode**: `{report.retrieval_mode.upper()}` (Top-{report.top_k})",
            f"- **Execution Mode**: `{'LIVE GEMINI' if report.is_live else 'OFFLINE MOCK'}`",
            f"- **Provider / Model**: `{report.provider_name}` / `{report.model_name}`",
            f"- **Duration**: `{report.total_duration_seconds}s` ({report.evaluated_cases}/{report.total_cases} cases evaluated)",
            f"- **Total Prompt Tokens**: `~{report.total_prompt_tokens}`",
            f"- **Overall Quality Gate**: {status_badge}",
            f"- **Hard Security & Integrity Gates**: {hard_badge}",
            "",
            "---",
            "",
            "## 1. Quality Gates & Policy Enforcement",
            "",
            "| Gate Metric | Type | Threshold | Actual | Result | Description |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ]

        for g in report.quality_gate.gate_details:
            g_type = "Hard Gate" if g.is_hard_gate else "Provisional"
            res = "PASSED ✅" if g.passed else "FAILED ❌"
            thresh_str = f"<= {g.threshold:.2f}" if "unnecessary" in g.gate_name else f">= {g.threshold:.2f}"
            lines.append(
                f"| `{g.gate_name}` | {g_type} | `{thresh_str}` | `{g.actual_value:.4f}` | {res} | {g.description} |"
            )

        # Macro Metrics Table
        lines.extend([
            "",
            "---",
            "",
            "## 2. Macro Aggregate Metrics",
            "",
            "| Metric | Score | Explanation |",
            "| :--- | :---: | :--- |",
        ])

        m = report.macro_metrics
        lines.append(f"| **File Recall@{report.top_k}** | `{m.get('file_recall_at_k', 0.0):.4f}` | Expected files retrieved in top-{report.top_k} (answerable queries) |")
        lines.append(f"| **File MRR** | `{m.get('file_mrr', 0.0):.4f}` | Mean reciprocal rank of first expected relevant file |")
        lines.append(f"| **File Hit Rate** | `{m.get('file_hit_rate', 0.0):.4f}` | Cases finding at least one relevant file |")
        lines.append(f"| **Citation Validity Rate** | `{m.get('citation_validity_rate', 0.0):.4f}` | Citations matching actual chunks in assembled context |")
        lines.append(f"| **Citation Boundary Correctness** | `{m.get('citation_boundary_rate', 0.0):.4f}` | Cited lines strictly within chunk start/end boundaries |")
        lines.append(f"| **Abstention Recall (TN / (TN+FP))** | `{m.get('abstention_recall', 0.0):.4f}` | Zero hallucination on unanswerable queries |")
        lines.append(f"| **Unnecessary Abstention Rate (FN / (TP+FN))** | `{m.get('unnecessary_abstention_rate', 0.0):.4f}` | False refusal rate on answerable queries |")
        lines.append(f"| **Answerability Accuracy** | `{m.get('answerability_accuracy', 0.0):.4f}` | Correct classification of answerability (TP + TN) / Total |")

        conf = m.get("confusion_matrix", {})
        lines.extend([
            "",
            f"**Abstention Confusion Matrix**: `TP={conf.get('TP', 0)}`, `TN={conf.get('TN', 0)}`, `FP={conf.get('FP', 0)} (Missed Abstentions)`, `FN={conf.get('FN', 0)} (False Refusals)`",
            "",
            "---",
            "",
            "## 3. Category Breakdown",
            "",
            "| Category | Cases | Recall | MRR | Hit Rate | Citation Validity | Abstention Acc | Fact Coverage (Smoke) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for cat_name, cat in report.category_summaries.items():
            rec_str = f"{cat.avg_file_recall:.4f}" if cat.avg_file_recall is not None else "N/A"
            lines.append(
                f"| `{cat_name}` | {cat.evaluated_cases}/{cat.total_cases} | `{rec_str}` | `{cat.avg_file_mrr:.4f}` | `{cat.avg_file_hit_rate:.4f}` | `{cat.citation_validity_rate:.4f}` | `{cat.abstention_accuracy:.4f}` | `{cat.fact_coverage_score:.4f}` |"
            )

        # Diagnostic Section: Failures and Skips
        if report.quality_gate.failing_cases:
            lines.extend([
                "",
                "---",
                "",
                "## 4. Failing Cases & Diagnostics",
                "",
            ])
            for fc in report.quality_gate.failing_cases:
                lines.append(f"### Case `{fc['case_id']}` (`{fc['category']}`)")
                for f_item in fc["failures"]:
                    lines.append(f"- ⚠️ {f_item}")
                lines.append("")

        if report.quality_gate.skipped_cases:
            lines.extend([
                "",
                "---",
                "",
                "## 5. Skipped / Errored Cases",
                "",
            ])
            for sc in report.quality_gate.skipped_cases:
                lines.append(f"- Case `{sc.get('case_id', 'unknown')}`: `{sc.get('reason', 'unknown')}`")

        return "\n".join(lines) + "\n"

    @classmethod
    def save_reports(
        cls,
        report: EvaluationRunReport,
        json_path: Optional[Path | str] = None,
        markdown_path: Optional[Path | str] = None,
    ) -> None:
        """Save JSON and/or Markdown report artifacts."""
        if json_path:
            jp = Path(json_path).resolve()
            jp.parent.mkdir(parents=True, exist_ok=True)
            jp.write_text(cls.to_json(report), encoding="utf-8")

        if markdown_path:
            mp = Path(markdown_path).resolve()
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(cls.to_markdown(report), encoding="utf-8")
