"""Data models and report schemas for agentic repair workflows."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from codebase_assistant.agent.diagnostics import DiagnosticResult


class TaskPlan(BaseModel):
    """Structured execution plan for a developer repair or feature task."""
    task: str
    analysis: str = ""
    target_files: List[str] = Field(default_factory=list)
    test_files: List[str] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "### Execution Plan",
            f"- **Analysis**: {self.analysis or 'Direct task execution'}",
            f"- **Target Files**: {', '.join(self.target_files) if self.target_files else 'None identified'}",
            f"- **Associated Tests**: {', '.join(self.test_files) if self.test_files else 'None identified'}",
            "- **Planned Steps**:",
        ]
        for idx, step in enumerate(self.steps or ["Inspect codebase", "Propose fix", "Run verification"], start=1):
            lines.append(f"  {idx}. {step}")

        lines.append("- **Success Criteria**:")
        for idx, crit in enumerate(self.success_criteria or ["All tests pass cleanly without errors"], start=1):
            lines.append(f"  {idx}. {crit}")

        return "\n".join(lines)


class RepairAttempt(BaseModel):
    """Record of a single bounded repair attempt within the workflow."""
    attempt_number: int
    proposed_diff: str = ""
    ast_valid: bool = True
    syntax_errors: List[str] = Field(default_factory=list)
    patch_applied: bool = False
    test_command: str = ""
    test_passed: bool = False
    test_output: str = ""
    diagnostics: Optional[DiagnosticResult] = None
    duration_seconds: float = 0.0


class RepairWorkflowStatus(str, Enum):
    """Outcome status of an agentic repair workflow."""
    SUCCESS = "success"
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    STOPPED_BY_USER = "stopped_by_user"
    PLANNING_FAILED = "planning_failed"
    SYNTAX_FAILED = "syntax_failed"
    ERROR = "error"


class RepairWorkflowReport(BaseModel):
    """Comprehensive final audit report summarizing the end-to-end repair workflow."""
    workflow_id: str
    task: str
    repository_path: str
    status: RepairWorkflowStatus
    plan: TaskPlan
    attempts: List[RepairAttempt] = Field(default_factory=list)
    final_diff: Optional[str] = None
    modified_files: List[str] = Field(default_factory=list)
    retrieved_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    total_attempts: int = 0
    total_duration_seconds: float = 0.0
    summary: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return self.model_dump()

    def to_markdown(self) -> str:
        """Render complete audit report in GitHub-flavored markdown."""
        status_icons = {
            RepairWorkflowStatus.SUCCESS: "🟢 SUCCESS",
            RepairWorkflowStatus.MAX_RETRIES_EXCEEDED: "🔴 FAILED (Max Retries Exceeded)",
            RepairWorkflowStatus.STOPPED_BY_USER: "🟡 STOPPED (User Authorization Denied)",
            RepairWorkflowStatus.PLANNING_FAILED: "🔴 FAILED (Planning Infeasible)",
            RepairWorkflowStatus.SYNTAX_FAILED: "🔴 FAILED (Syntax Verification Failed)",
            RepairWorkflowStatus.ERROR: "🔴 ERROR",
        }
        status_str = status_icons.get(self.status, self.status.value.upper())

        lines = [
            f"# Agentic Repair Workflow Report: `{self.workflow_id}`",
            "",
            f"- **Task**: {self.task}",
            f"- **Status**: {status_str}",
            f"- **Total Attempts**: {self.total_attempts}",
            f"- **Duration**: {self.total_duration_seconds:.2f} seconds",
            f"- **Target Repository**: `{self.repository_path}`",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            "",
            self.summary or "Repair workflow completed.",
            "",
            "---",
            "",
            "## 2. Task Plan",
            "",
            self.plan.to_markdown(),
            "",
            "---",
            "",
            "## 3. Repair Attempts & Test Diagnostic History",
            "",
        ]

        if not self.attempts:
            lines.append("No repair attempts were recorded.")
        else:
            for att in self.attempts:
                pass_label = "✅ PASSED" if att.test_passed else "❌ FAILED"
                lines.extend([
                    f"### Attempt #{att.attempt_number} ({att.duration_seconds:.2f}s) — {pass_label}",
                    f"- **AST Valid**: {'Yes' if att.ast_valid else 'No: ' + '; '.join(att.syntax_errors)}",
                    f"- **Patch Applied**: {'Yes' if att.patch_applied else 'No'}",
                    f"- **Test Command**: `{att.test_command}`",
                ])

                if att.diagnostics and not att.test_passed:
                    lines.extend([
                        f"- **Error Type**: `{att.diagnostics.error_type}`",
                        f"- **Failure Summary**: {att.diagnostics.failure_summary}",
                        f"- **Hypothesized Cause**: {att.diagnostics.hypothesized_cause}",
                        f"- **Recommended Direction**: {att.diagnostics.recommended_fix_direction}",
                    ])
                    if att.diagnostics.stack_trace_snippet:
                        lines.extend([
                            "```text",
                            att.diagnostics.stack_trace_snippet.strip(),
                            "```",
                        ])

                if att.proposed_diff:
                    lines.extend([
                        "",
                        "<details><summary>View Proposed Diff (Attempt #{})</summary>".format(att.attempt_number),
                        "",
                        "```diff",
                        att.proposed_diff.strip(),
                        "```",
                        "</details>",
                        "",
                    ])

        lines.extend([
            "---",
            "",
            "## 4. Final Patch Proposal",
            "",
        ])

        if self.final_diff:
            lines.extend([
                f"**Modified Files**: {', '.join(self.modified_files) if self.modified_files else 'None'}",
                "",
                "```diff",
                self.final_diff.strip(),
                "```",
            ])
        else:
            lines.append("No verified patch proposal was generated.")

        lines.extend([
            "",
            "---",
            "",
            "## 5. Retrieved Evidence & Citations",
            "",
        ])

        if not self.retrieved_evidence:
            lines.append("No context chunks retrieved.")
        else:
            for idx, ev in enumerate(self.retrieved_evidence[:8], start=1):
                sym = ev.get("symbol_name") or ev.get("chunk_type", "chunk")
                file_path = ev.get("file_path", "unknown")
                lines.append(f"- **[{idx}] `{file_path}`** (`{sym}`): {ev.get('summary', '')[:120]}")

        return "\n".join(lines)
