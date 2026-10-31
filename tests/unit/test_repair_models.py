"""Unit tests for repair data models and report formatting."""

import unittest

from codebase_assistant.agent.diagnostics import DiagnosticResult
from codebase_assistant.agent.repair_models import (
    RepairAttempt,
    RepairWorkflowReport,
    RepairWorkflowStatus,
    TaskPlan,
)


class TestRepairModels(unittest.TestCase):
    """Test suite verifying TaskPlan, RepairAttempt, and RepairWorkflowReport schemas."""

    def test_task_plan_markdown_rendering(self):
        plan = TaskPlan(
            task="Fix divide by zero bug",
            analysis="Add check for zero divisor in calc.py",
            target_files=["src/calc.py"],
            test_files=["tests/unit/test_calc.py"],
            steps=["Inspect code", "Propose fix", "Run tests"],
            success_criteria=["Tests pass cleanly"],
        )
        md = plan.to_markdown()
        self.assertIn("### Execution Plan", md)
        self.assertIn("Fix divide by zero bug", plan.task)
        self.assertIn("src/calc.py", md)
        self.assertIn("tests/unit/test_calc.py", md)
        self.assertIn("1. Inspect code", md)
        self.assertIn("1. Tests pass cleanly", md)

    def test_repair_workflow_report_success_markdown(self):
        plan = TaskPlan(
            task="Add multiply function",
            target_files=["src/calc.py"],
        )
        attempt = RepairAttempt(
            attempt_number=1,
            proposed_diff="--- a/src/calc.py\n+++ b/src/calc.py\n@@ -1,1 +1,3 @@\n+def mul(a, b): return a * b\n",
            ast_valid=True,
            patch_applied=True,
            test_command="python -m unittest tests/unit/test_calc.py",
            test_passed=True,
            test_output="OK",
            duration_seconds=1.23,
        )
        report = RepairWorkflowReport(
            workflow_id="repair-abc12345",
            task="Add multiply function",
            repository_path="/repo",
            status=RepairWorkflowStatus.SUCCESS,
            plan=plan,
            attempts=[attempt],
            final_diff=attempt.proposed_diff,
            modified_files=["src/calc.py"],
            total_attempts=1,
            total_duration_seconds=2.45,
            summary="Task successfully resolved in 1 attempt.",
        )

        md = report.to_markdown()
        self.assertIn("# Agentic Repair Workflow Report: `repair-abc12345`", md)
        self.assertIn("🟢 SUCCESS", md)
        self.assertIn("## 1. Executive Summary", md)
        self.assertIn("## 2. Task Plan", md)
        self.assertIn("## 3. Repair Attempts & Test Diagnostic History", md)
        self.assertIn("Attempt #1", md)
        self.assertIn("## 4. Final Patch Proposal", md)
        self.assertIn("def mul(a, b)", md)

        # Dictionary conversion
        d = report.to_dict()
        self.assertEqual(d["status"], "success")
        self.assertEqual(d["total_attempts"], 1)

    def test_repair_workflow_report_failure_markdown(self):
        plan = TaskPlan(task="Fix broken function")
        diag = DiagnosticResult(
            passed=False,
            exit_code=1,
            failure_summary="AssertionError: 1 != 2",
            error_type="AssertionError",
            hypothesized_cause="Calculation mismatch",
            recommended_fix_direction="Check return statement",
        )
        attempt = RepairAttempt(
            attempt_number=1,
            test_passed=False,
            diagnostics=diag,
        )
        report = RepairWorkflowReport(
            workflow_id="repair-fail1234",
            task="Fix broken function",
            repository_path="/repo",
            status=RepairWorkflowStatus.MAX_RETRIES_EXCEEDED,
            plan=plan,
            attempts=[attempt],
            total_attempts=1,
            total_duration_seconds=5.0,
            summary="Max retries exceeded.",
        )
        md = report.to_markdown()
        self.assertIn("🔴 FAILED (Max Retries Exceeded)", md)
        self.assertIn("AssertionError", md)
        self.assertIn("Calculation mismatch", md)


if __name__ == "__main__":
    unittest.main()
