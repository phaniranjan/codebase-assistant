"""Agentic repair workflow coordinator for end-to-end task execution and bounded self-repair."""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from codebase_assistant.agent.diagnostics import (
    DiagnosticEngine,
    DiagnosticResult,
    sanitize_untrusted_text,
)
from codebase_assistant.agent.models import ToolCall, ToolType
from codebase_assistant.agent.permissions import PermissionEngine
from codebase_assistant.agent.policy import PolicyEngine, SecurityPolicyError
from codebase_assistant.agent.repair_models import (
    RepairAttempt,
    RepairWorkflowReport,
    RepairWorkflowStatus,
    TaskPlan,
)
from codebase_assistant.agent.tools import ToolRegistry
from codebase_assistant.generation.code_generator import CodeProposalGenerator
from codebase_assistant.generation.diff_engine import DiffEngine
from codebase_assistant.generation.test_generator import TestProposalGenerator
from codebase_assistant.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class RepairWorkflowCoordinator:
    """Orchestrates plan creation, context retrieval, patch application, sandboxed testing, and bounded repairs."""

    def __init__(
        self,
        repository_path: Path | str,
        retriever: Optional[HybridRetriever] = None,
        code_generator: Optional[CodeProposalGenerator] = None,
        test_generator: Optional[TestProposalGenerator] = None,
        gemini_client: Optional[Any] = None,
        policy_engine: Optional[PolicyEngine] = None,
        permission_engine: Optional[PermissionEngine] = None,
        tool_registry: Optional[ToolRegistry] = None,
        diagnostic_engine: Optional[DiagnosticEngine] = None,
        max_retries: int = 3,
        max_reasoning_steps: int = 15,
        command_timeout_seconds: float = 30.0,
        workflow_timeout_seconds: float = 120.0,
        token_budget: int = 8192,
    ) -> None:
        self.repository_path = Path(repository_path).resolve()
        self.retriever = retriever
        self.code_generator = code_generator
        self.test_generator = test_generator
        self.gemini_client = gemini_client
        self.policy_engine = policy_engine or PolicyEngine(self.repository_path)
        self.permission_engine = permission_engine or PermissionEngine()
        self.diagnostic_engine = diagnostic_engine or DiagnosticEngine()
        self.max_retries = max(1, max_retries)
        self.max_reasoning_steps = max(1, max_reasoning_steps)
        self.command_timeout_seconds = max(0.1, command_timeout_seconds)
        self.workflow_timeout_seconds = max(1.0, workflow_timeout_seconds)
        self.token_budget = max(100, token_budget)

        self.tool_registry = tool_registry or ToolRegistry(
            repository_path=self.repository_path,
            retriever=self.retriever,
            code_generator=self.code_generator,
            test_generator=self.test_generator,
            policy_engine=self.policy_engine,
            permission_engine=self.permission_engine,
            command_timeout_seconds=self.command_timeout_seconds,
        )

    def _count_tokens(self, text: str) -> int:
        """Count or estimate tokens consumed by text."""
        if not text:
            return 0
        if self.gemini_client and hasattr(self.gemini_client, "count_tokens"):
            try:
                return self.gemini_client.count_tokens(text)
            except Exception:
                pass
        # Robust fallback estimation (~4 chars per token)
        return max(1, len(text) // 4)

    def create_plan(
        self,
        task: str,
        target_file: Optional[str] = None,
        test_command: Optional[str] = None,
    ) -> TaskPlan:
        """Analyze developer task and generate a structured execution plan."""
        analysis = f"Analyze and execute task: '{task}'"
        target_files = [target_file] if target_file else []
        test_files: list[str] = []
        planning_error: Optional[str] = None

        # If LLM client is available, request plan decomposition
        if self.gemini_client:
            prompt = (
                "You are an expert software engineer and autonomous code repair agent.\n"
                f"Analyze this task for repository '{self.repository_path.name}':\n"
                f"Task: {task}\n\n"
                "Return a strict JSON object with these keys:\n"
                "{\n"
                '  "analysis": "concise architectural understanding of the task",\n'
                '  "target_files": ["candidate relative paths to modify"],\n'
                '  "test_files": ["candidate test file paths"],\n'
                '  "steps": ["step 1", "step 2", ...],\n'
                '  "success_criteria": ["criterion 1", ...]\n'
                "}\n"
                "Return ONLY valid JSON."
            )
            try:
                resp = self.gemini_client.generate(prompt)
                raw_text = resp.text.strip()
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json", 1)[1].split("```", 1)[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```", 1)[1].split("```", 1)[0].strip()

                parsed = json.loads(raw_text)
                return TaskPlan(
                    task=task,
                    analysis=parsed.get("analysis", analysis),
                    target_files=parsed.get("target_files", target_files),
                    test_files=parsed.get("test_files", test_files),
                    steps=parsed.get(
                        "steps",
                        [
                            "Retrieve relevant context and symbols",
                            "Generate unified diff patch proposal",
                            "Validate syntax via AST parsing",
                            "Apply patch to workspace",
                            "Execute sandboxed test runner",
                        ],
                    ),
                    success_criteria=parsed.get(
                        "success_criteria",
                        [
                            "Code passes deterministic AST validation",
                            "All target tests pass with exit code 0",
                        ],
                    ),
                )
            except Exception as e:
                planning_error = f"LLM planning failed ({type(e).__name__}: {str(e)}). Using deterministic fallback plan."
                logger.warning(planning_error)

        # Fallback heuristic planning
        steps = [
            f"Locate code chunks and symbols matching: '{task}'",
            "Generate context-aware unified diff proposal",
            "Deterministically verify AST syntax",
            "Apply patch to repository workspace",
            "Execute sandboxed test suite to verify behavior",
        ]
        criteria = [
            "All targeted unit tests pass cleanly",
            "Zero syntax errors or regressions introduced",
        ]
        return TaskPlan(
            task=task,
            analysis=analysis,
            target_files=target_files,
            test_files=test_files,
            steps=steps,
            success_criteria=criteria,
            planning_error=planning_error,
        )

    def run(
        self,
        task: str,
        target_file: Optional[str] = None,
        test_command: Optional[str] = None,
    ) -> RepairWorkflowReport:
        """Execute an end-to-end agentic repair workflow with bounded diagnostic retries."""
        start_time = time.time()
        workflow_id = f"repair-{uuid.uuid4().hex[:8]}"
        logger.info("Initiating repair workflow %s for task: '%s'", workflow_id, task)

        cumulative_tokens: int = 0
        cumulative_steps: int = 0
        rollback_errors: List[str] = []
        rolled_back_files: List[str] = []
        rollback_performed: bool = False

        # 1. Plan
        cumulative_steps += 1
        plan = self.create_plan(task, target_file=target_file, test_command=test_command)
        if self.gemini_client:
            cumulative_tokens += self._count_tokens(task) + 200

        # Check reasoning steps and token limits after planning
        if cumulative_steps > self.max_reasoning_steps:
            return RepairWorkflowReport(
                workflow_id=workflow_id,
                task=task,
                repository_path=str(self.repository_path),
                status=RepairWorkflowStatus.MAX_STEPS_EXCEEDED,
                plan=plan,
                total_steps=cumulative_steps,
                tokens_consumed=cumulative_tokens,
                summary=f"Workflow exceeded cumulative reasoning steps limit ({cumulative_steps} > {self.max_reasoning_steps}).",
                total_duration_seconds=time.time() - start_time,
            )

        if cumulative_tokens > self.token_budget:
            return RepairWorkflowReport(
                workflow_id=workflow_id,
                task=task,
                repository_path=str(self.repository_path),
                status=RepairWorkflowStatus.BUDGET_EXCEEDED,
                plan=plan,
                total_steps=cumulative_steps,
                tokens_consumed=cumulative_tokens,
                summary=f"Workflow exceeded token budget limit ({cumulative_tokens} > {self.token_budget}).",
                total_duration_seconds=time.time() - start_time,
            )

        # 2. Context Retrieval & Target Discovery
        cumulative_steps += 1
        retrieved_evidence: List[Dict[str, Any]] = []
        if self.retriever:
            try:
                results = self.retriever.search(task, top_k=5, mode="hybrid")
                for r in results:
                    c = r.chunk
                    retrieved_evidence.append(
                        {
                            "file_path": c.file_path,
                            "symbol_name": c.symbol_name or c.parent_symbol or c.chunk_type.value,
                            "chunk_type": c.chunk_type.value,
                            "summary": c.content.strip().splitlines()[0] if c.content else "",
                        }
                    )
                # If target_file was not provided, pick top retrieved file
                if not target_file and results:
                    target_file = results[0].chunk.file_path
                    if target_file not in plan.target_files:
                        plan.target_files.insert(0, target_file)
            except Exception as e:
                logger.warning("Context retrieval failed: %s", e)

        # Validate target file containment
        if target_file:
            try:
                self.policy_engine.validate_file_access(target_file)
            except (SecurityPolicyError, ValueError) as pe:
                return RepairWorkflowReport(
                    workflow_id=workflow_id,
                    task=task,
                    repository_path=str(self.repository_path),
                    status=RepairWorkflowStatus.PLANNING_FAILED,
                    plan=plan,
                    total_steps=cumulative_steps,
                    tokens_consumed=cumulative_tokens,
                    summary=f"Path security violation: target file '{target_file}' violates workspace boundary: {pe}",
                    total_duration_seconds=time.time() - start_time,
                )

        if not target_file:
            return RepairWorkflowReport(
                workflow_id=workflow_id,
                task=task,
                repository_path=str(self.repository_path),
                status=RepairWorkflowStatus.PLANNING_FAILED,
                plan=plan,
                total_steps=cumulative_steps,
                tokens_consumed=cumulative_tokens,
                summary="Failed to identify valid target file to modify within workspace boundaries.",
                total_duration_seconds=time.time() - start_time,
            )

        # Determine effective test command
        effective_test_cmd = test_command
        if not effective_test_cmd:
            cumulative_steps += 1
            # Discover related tests
            res_tests = self.tool_registry.execute_tool(
                ToolCall(tool_name="find_related_tests", arguments={"symbol_or_path": target_file})
            )
            stem = Path(target_file).stem.replace("test_", "")
            candidate_test = f"tests/unit/test_{stem}.py"
            if (self.repository_path / candidate_test).is_file():
                effective_test_cmd = f"python -m unittest {candidate_test}"
            else:
                effective_test_cmd = "python -m unittest discover tests"

        # Validate test command upfront through PolicyEngine
        try:
            self.policy_engine.validate_test_command(effective_test_cmd)
        except SecurityPolicyError as e:
            return RepairWorkflowReport(
                workflow_id=workflow_id,
                task=task,
                repository_path=str(self.repository_path),
                status=RepairWorkflowStatus.PLANNING_FAILED,
                plan=plan,
                total_steps=cumulative_steps,
                tokens_consumed=cumulative_tokens,
                summary=f"Security policy rejection for test command '{effective_test_cmd}': {e}",
                total_duration_seconds=time.time() - start_time,
            )

        # Track original contents of all touched files for atomic rollback
        original_contents: Dict[Path, Optional[str]] = {}
        abs_target = self.policy_engine.validate_file_access(target_file)
        if abs_target.is_file():
            original_contents[abs_target] = abs_target.read_text(encoding="utf-8")
        else:
            original_contents[abs_target] = None

        any_patch_applied = False

        def _backup_targets_for_diff(unified_diff: str) -> None:
            """Ensure all file paths targeted in a proposed diff are backed up before applying."""
            try:
                targets = self.policy_engine.validate_patch_targets(unified_diff)
                for t in targets:
                    if t not in original_contents:
                        if t.is_file():
                            original_contents[t] = t.read_text(encoding="utf-8")
                        else:
                            original_contents[t] = None
            except Exception as backup_err:
                logger.debug("Pre-backup target check: %s", backup_err)

        def _restore_originals() -> List[str]:
            """Restore all modified files to exact baseline state. Returns list of rollback errors if any."""
            nonlocal any_patch_applied, rollback_performed
            if not any_patch_applied:
                return []

            rollback_performed = True
            errors: List[str] = []
            restored: List[str] = []

            for path, orig_text in list(original_contents.items()):
                try:
                    if orig_text is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(orig_text, encoding="utf-8")
                        rel = path.relative_to(self.repository_path).as_posix()
                        if rel not in restored:
                            restored.append(rel)
                    elif path.exists():
                        path.unlink()
                        rel = path.relative_to(self.repository_path).as_posix()
                        if rel not in restored:
                            restored.append(rel)
                except Exception as ex:
                    msg = f"Failed to restore '{path}': {ex}"
                    logger.error(msg)
                    errors.append(msg)

            if not errors:
                any_patch_applied = False
                for r in restored:
                    if r not in rolled_back_files:
                        rolled_back_files.append(r)

            return errors

        attempts: List[RepairAttempt] = []
        workflow_status = RepairWorkflowStatus.MAX_RETRIES_EXCEEDED
        final_diff: Optional[str] = None
        modified_files: List[str] = []
        previous_diagnostics: Optional[DiagnosticResult] = None
        previous_diff: Optional[str] = None

        try:
            # 3. Bounded Repair Loop
            for attempt_idx in range(1, self.max_retries + 1):
                attempt_start = time.time()
                logger.info("Executing repair attempt %d of %d", attempt_idx, self.max_retries)

                # Check cumulative workflow execution timeout before attempt
                if (time.time() - start_time) >= self.workflow_timeout_seconds:
                    workflow_status = RepairWorkflowStatus.TIMEOUT
                    rb_errs = _restore_originals()
                    if rb_errs:
                        rollback_errors.extend(rb_errs)
                        workflow_status = RepairWorkflowStatus.ROLLBACK_FAILED
                    break

                # Check cumulative reasoning steps limit
                cumulative_steps += 1
                if cumulative_steps > self.max_reasoning_steps:
                    workflow_status = RepairWorkflowStatus.MAX_STEPS_EXCEEDED
                    rb_errs = _restore_originals()
                    if rb_errs:
                        rollback_errors.extend(rb_errs)
                        workflow_status = RepairWorkflowStatus.ROLLBACK_FAILED
                    break

                # A. Generate Code Proposal
                if attempt_idx == 1 or not previous_diagnostics:
                    prop_task = task
                else:
                    # Formulate feedback prompt for self-repair with sanitized diagnostics
                    clean_diff = sanitize_untrusted_text(previous_diff or "", max_chars=2500)
                    prop_task = (
                        f"{task}\n\n"
                        f"Previous repair attempt #{attempt_idx - 1} failed test verification.\n"
                        f"--- Previous Diff ---\n{clean_diff}\n\n"
                        f"{previous_diagnostics.to_feedback_prompt()}\n\n"
                        f"Please generate a corrected minimal diff that fixes the diagnosed error without regressions."
                    )

                if not self.code_generator:
                    return RepairWorkflowReport(
                        workflow_id=workflow_id,
                        task=task,
                        repository_path=str(self.repository_path),
                        status=RepairWorkflowStatus.ERROR,
                        plan=plan,
                        total_steps=cumulative_steps,
                        tokens_consumed=cumulative_tokens,
                        summary="Code proposal generator is not configured.",
                        total_duration_seconds=time.time() - start_time,
                    )

                prop_res = self.code_generator.propose(task=prop_task, target_file=target_file)
                cumulative_tokens += self._count_tokens(prop_task) + 600

                # Check token budget after proposal generation
                if cumulative_tokens > self.token_budget:
                    workflow_status = RepairWorkflowStatus.BUDGET_EXCEEDED
                    rb_errs = _restore_originals()
                    if rb_errs:
                        rollback_errors.extend(rb_errs)
                        workflow_status = RepairWorkflowStatus.ROLLBACK_FAILED
                    break

                if not prop_res.proposals:
                    attempts.append(
                        RepairAttempt(
                            attempt_number=attempt_idx,
                            proposed_diff="",
                            ast_valid=False,
                            syntax_errors=[prop_res.verification_summary],
                            duration_seconds=time.time() - attempt_start,
                        )
                    )
                    continue

                proposal = prop_res.proposals[0]
                diff_str = proposal.unified_diff

                if not proposal.ast_valid:
                    attempts.append(
                        RepairAttempt(
                            attempt_number=attempt_idx,
                            proposed_diff=diff_str,
                            ast_valid=False,
                            syntax_errors=proposal.syntax_errors,
                            duration_seconds=time.time() - attempt_start,
                        )
                    )
                    continue

                # Pre-backup all files that this diff could modify or create
                _backup_targets_for_diff(diff_str)

                # B. Apply Patch in Isolation
                cumulative_steps += 1
                if cumulative_steps > self.max_reasoning_steps:
                    workflow_status = RepairWorkflowStatus.MAX_STEPS_EXCEEDED
                    rb_errs = _restore_originals()
                    if rb_errs:
                        rollback_errors.extend(rb_errs)
                        workflow_status = RepairWorkflowStatus.ROLLBACK_FAILED
                    break

                apply_call = ToolCall(tool_name="apply_patch", arguments={"unified_diff": diff_str})
                apply_res = self.tool_registry.execute_tool(apply_call)

                if not apply_res.is_authorized:
                    rb_errs = _restore_originals()
                    if rb_errs:
                        rollback_errors.extend(rb_errs)
                        workflow_status = RepairWorkflowStatus.ROLLBACK_FAILED
                    else:
                        workflow_status = RepairWorkflowStatus.STOPPED_BY_USER
                    attempts.append(
                        RepairAttempt(
                            attempt_number=attempt_idx,
                            proposed_diff=diff_str,
                            ast_valid=True,
                            patch_applied=False,
                            rolled_back=False,
                            duration_seconds=time.time() - attempt_start,
                        )
                    )
                    break

                if not apply_res.success:
                    attempts.append(
                        RepairAttempt(
                            attempt_number=attempt_idx,
                            proposed_diff=diff_str,
                            ast_valid=True,
                            patch_applied=False,
                            rolled_back=False,
                            syntax_errors=[apply_res.error or "Patch application failed"],
                            duration_seconds=time.time() - attempt_start,
                        )
                    )
                    continue

                patch_applied = True
                any_patch_applied = True
                modified_files = [target_file]

                # C. Execute Sandboxed Tests
                cumulative_steps += 1
                if (time.time() - start_time) >= self.workflow_timeout_seconds:
                    workflow_status = RepairWorkflowStatus.TIMEOUT
                    rb_errs = _restore_originals()
                    if rb_errs:
                        rollback_errors.extend(rb_errs)
                        workflow_status = RepairWorkflowStatus.ROLLBACK_FAILED
                    break

                test_call = ToolCall(
                    tool_name="run_selected_tests", arguments={"command": effective_test_cmd}
                )
                test_res = self.tool_registry.execute_tool(test_call)

                if not test_res.is_authorized:
                    rb_errs = _restore_originals()
                    if rb_errs:
                        rollback_errors.extend(rb_errs)
                        workflow_status = RepairWorkflowStatus.ROLLBACK_FAILED
                    else:
                        workflow_status = RepairWorkflowStatus.STOPPED_BY_USER
                    attempts.append(
                        RepairAttempt(
                            attempt_number=attempt_idx,
                            proposed_diff=diff_str,
                            ast_valid=True,
                            patch_applied=True,
                            test_command=effective_test_cmd,
                            rolled_back=True,
                            duration_seconds=time.time() - attempt_start,
                        )
                    )
                    break

                # D. Analyze Diagnostics
                cumulative_steps += 1
                diagnostics = self.diagnostic_engine.analyze(test_res.output)
                test_passed = diagnostics.passed and test_res.success

                attempt_record = RepairAttempt(
                    attempt_number=attempt_idx,
                    proposed_diff=diff_str,
                    ast_valid=True,
                    patch_applied=patch_applied,
                    test_command=effective_test_cmd,
                    test_passed=test_passed,
                    test_output=test_res.output,
                    diagnostics=diagnostics,
                    rolled_back=not test_passed,
                    duration_seconds=time.time() - attempt_start,
                )
                attempts.append(attempt_record)

                if test_passed:
                    # Success achieved!
                    workflow_status = RepairWorkflowStatus.SUCCESS
                    final_diff = diff_str
                    break
                else:
                    # Test failed: prepare diagnostics for next iteration and revert workspace
                    previous_diagnostics = diagnostics
                    previous_diff = diff_str
                    rb_errs = _restore_originals()
                    if rb_errs:
                        rollback_errors.extend(rb_errs)
                        workflow_status = RepairWorkflowStatus.ROLLBACK_FAILED
                        attempt_record.rolled_back = False
                        break

        except Exception as e:
            logger.exception("Unexpected error during repair workflow: %s", e)
            rb_errs = _restore_originals()
            if rb_errs:
                rollback_errors.extend(rb_errs)
                workflow_status = RepairWorkflowStatus.ROLLBACK_FAILED
            else:
                workflow_status = RepairWorkflowStatus.ERROR

        # If workflow did not succeed, ensure workspace is restored cleanly
        if workflow_status != RepairWorkflowStatus.SUCCESS:
            rb_errs = _restore_originals()
            if rb_errs:
                rollback_errors.extend(rb_errs)
                workflow_status = RepairWorkflowStatus.ROLLBACK_FAILED
            modified_files = []

        total_duration = time.time() - start_time
        summary = self._generate_summary(workflow_status, attempts, final_diff, rollback_errors)

        return RepairWorkflowReport(
            workflow_id=workflow_id,
            task=task,
            repository_path=str(self.repository_path),
            status=workflow_status,
            plan=plan,
            attempts=attempts,
            final_diff=final_diff,
            modified_files=modified_files,
            rolled_back_files=rolled_back_files,
            rollback_performed=rollback_performed,
            rollback_success=len(rollback_errors) == 0,
            rollback_errors=rollback_errors,
            retrieved_evidence=retrieved_evidence,
            total_attempts=len(attempts),
            total_steps=cumulative_steps,
            tokens_consumed=cumulative_tokens,
            planning_error=plan.planning_error,
            total_duration_seconds=total_duration,
            summary=summary,
        )

    @staticmethod
    def _generate_summary(
        status: RepairWorkflowStatus,
        attempts: List[RepairAttempt],
        final_diff: Optional[str],
        rollback_errors: Optional[List[str]] = None,
    ) -> str:
        if status == RepairWorkflowStatus.ROLLBACK_FAILED:
            err_details = "; ".join(
                rollback_errors or ["Disk write permission failure during file restoration"]
            )
            return (
                f"CRITICAL ROLLBACK FAILURE: Repair workflow failed and was unable to restore the original "
                f"workspace state. Errors encountered: {err_details}. The workspace may be in an inconsistent state."
            )
        if status == RepairWorkflowStatus.SUCCESS:
            return (
                f"Successfully completed task in {len(attempts)} attempt(s). "
                f"The proposed patch passed deterministic AST validation and all sandboxed test checks."
            )
        if status == RepairWorkflowStatus.STOPPED_BY_USER:
            return "Repair workflow halted because user denied authorization for consequential action. Changes rolled back."
        if status == RepairWorkflowStatus.BUDGET_EXCEEDED:
            return "Repair workflow halted because the cumulative context token budget was exceeded. Changes rolled back."
        if status == RepairWorkflowStatus.MAX_STEPS_EXCEEDED:
            return "Repair workflow halted because the cumulative reasoning steps limit was exceeded. Changes rolled back."
        if status == RepairWorkflowStatus.TIMEOUT:
            return "Repair workflow halted because execution exceeded the cumulative workflow timeout. Changes rolled back."
        if status == RepairWorkflowStatus.MAX_RETRIES_EXCEEDED:
            diag_msgs = [
                a.diagnostics.failure_summary
                for a in attempts
                if a.diagnostics and a.diagnostics.failure_summary
            ]
            latest_error = f" Latest failure: {diag_msgs[-1]}" if diag_msgs else ""
            return (
                f"Exceeded maximum repair retry limit ({len(attempts)} attempts).{latest_error} "
                f"Workspace changes have been safely reverted to the clean baseline state."
            )
        if status == RepairWorkflowStatus.SYNTAX_FAILED:
            return "Failed to produce a proposal that passed AST syntax verification."
        return "Repair workflow terminated due to an error."
