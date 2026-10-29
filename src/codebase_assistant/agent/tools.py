"""Tool definitions and registry for Agent execution."""

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from codebase_assistant.agent.models import (
    AuthorizationRequest,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolType,
)
from codebase_assistant.agent.permissions import PermissionEngine
from codebase_assistant.agent.policy import PolicyEngine, SecurityPolicyError
from codebase_assistant.generation.code_generator import CodeProposalGenerator
from codebase_assistant.generation.diff_engine import DiffEngine
from codebase_assistant.generation.models import ProposalResult
from codebase_assistant.generation.test_generator import TestProposalGenerator
from codebase_assistant.ingestion.models import CodeChunk
from codebase_assistant.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Manages agent tools, parameter validation, security checks, and dispatching."""

    def __init__(
        self,
        repository_path: Path | str,
        retriever: Optional[HybridRetriever] = None,
        code_generator: Optional[CodeProposalGenerator] = None,
        test_generator: Optional[TestProposalGenerator] = None,
        policy_engine: Optional[PolicyEngine] = None,
        permission_engine: Optional[PermissionEngine] = None,
        command_timeout_seconds: float = 30.0,
    ) -> None:
        self.repository_path = Path(repository_path).resolve()
        self.retriever = retriever
        self.code_generator = code_generator
        self.test_generator = test_generator
        self.policy_engine = policy_engine or PolicyEngine(self.repository_path)
        self.permission_engine = permission_engine or PermissionEngine()
        self.command_timeout_seconds = command_timeout_seconds

        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable[[Dict[str, Any]], str]] = {}
        self.collected_proposals: List[ProposalResult] = []

        self._register_default_tools()

    def register_tool(
        self,
        definition: ToolDefinition,
        handler: Callable[[Dict[str, Any]], str],
    ) -> None:
        """Register a tool definition and its execution handler."""
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler

    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Return list of all registered tool specifications."""
        return list(self._tools.values())

    def execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a tool call with policy verification and permission checks."""
        start_time = time.time()
        name = call.tool_name
        args = call.arguments

        if name not in self._tools:
            return ToolResult(
                tool_name=name,
                success=False,
                output="",
                error=f"Unknown tool '{name}'. Available tools: {list(self._tools.keys())}",
                is_authorized=True,
                execution_duration=time.time() - start_time,
            )

        definition = self._tools[name]
        handler = self._handlers[name]

        try:
            output = handler(args)
            return ToolResult(
                tool_name=name,
                success=True,
                output=output,
                is_authorized=True,
                execution_duration=time.time() - start_time,
            )
        except SecurityPolicyError as e:
            return ToolResult(
                tool_name=name,
                success=False,
                output="",
                error=f"Security Policy Rejection: {str(e)}",
                is_authorized=True,
                execution_duration=time.time() - start_time,
            )
        except PermissionError as e:
            return ToolResult(
                tool_name=name,
                success=False,
                output="",
                error=str(e),
                is_authorized=False,
                execution_duration=time.time() - start_time,
            )
        except Exception as e:
            logger.exception("Error executing tool %s: %s", name, e)
            return ToolResult(
                tool_name=name,
                success=False,
                output="",
                error=f"Execution error: {str(e)}",
                is_authorized=True,
                execution_duration=time.time() - start_time,
            )

    def _register_default_tools(self) -> None:
        """Register the 8 standard tools defined in AGENTS.md."""

        # 1. search_codebase
        self.register_tool(
            ToolDefinition(
                name="search_codebase",
                description="Search the codebase using hybrid vector and keyword search to find relevant functions, classes, and logic.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query or code symbol to search."},
                        "top_k": {"type": "integer", "description": "Number of chunks to return (default: 5).", "default": 5},
                    },
                    "required": ["query"],
                },
                tool_type=ToolType.READ_ONLY,
                requires_authorization=False,
            ),
            self._handle_search_codebase,
        )

        # 2. read_source_file
        self.register_tool(
            ToolDefinition(
                name="read_source_file",
                description="Read source file contents within the repository. Supports line range slicing.",
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Relative file path inside the repository."},
                        "start_line": {"type": "integer", "description": "Optional 1-based start line."},
                        "end_line": {"type": "integer", "description": "Optional 1-based end line."},
                    },
                    "required": ["file_path"],
                },
                tool_type=ToolType.READ_ONLY,
                requires_authorization=False,
            ),
            self._handle_read_source_file,
        )

        # 3. find_related_tests
        self.register_tool(
            ToolDefinition(
                name="find_related_tests",
                description="Find test files and test functions related to a given source file or symbol.",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol_or_path": {"type": "string", "description": "Source file path or symbol name to find tests for."},
                    },
                    "required": ["symbol_or_path"],
                },
                tool_type=ToolType.READ_ONLY,
                requires_authorization=False,
            ),
            self._handle_find_related_tests,
        )

        # 4. inspect_dependencies
        self.register_tool(
            ToolDefinition(
                name="inspect_dependencies",
                description="Inspect dependency manifests (e.g. pyproject.toml, requirements.txt) in the repository.",
                parameters={"type": "object", "properties": {}},
                tool_type=ToolType.READ_ONLY,
                requires_authorization=False,
            ),
            self._handle_inspect_dependencies,
        )

        # 5. generate_code_proposal
        self.register_tool(
            ToolDefinition(
                name="generate_code_proposal",
                description="Generate a context-aware unified diff code proposal with deterministic AST verification.",
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Specific code modification or creation task."},
                        "target_file": {"type": "string", "description": "Relative path of file to create or modify."},
                    },
                    "required": ["task"],
                },
                tool_type=ToolType.PROPOSAL,
                requires_authorization=False,
            ),
            self._handle_generate_code_proposal,
        )

        # 6. generate_test_proposal
        self.register_tool(
            ToolDefinition(
                name="generate_test_proposal",
                description="Generate a targeted unit test suite in unified diff format with AST verification.",
                parameters={
                    "type": "object",
                    "properties": {
                        "source_file": {"type": "string", "description": "Source file path to generate unit tests for."},
                        "framework": {"type": "string", "description": "Optional test framework ('pytest' or 'unittest')."},
                    },
                    "required": ["source_file"],
                },
                tool_type=ToolType.PROPOSAL,
                requires_authorization=False,
            ),
            self._handle_generate_test_proposal,
        )

        # 7. run_selected_tests
        self.register_tool(
            ToolDefinition(
                name="run_selected_tests",
                description="Execute an authorized test command (e.g. pytest, python -m unittest). Requires developer approval.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Test command to execute (e.g., 'pytest tests/test_calc.py')."},
                    },
                    "required": ["command"],
                },
                tool_type=ToolType.CONSEQUENTIAL,
                requires_authorization=True,
            ),
            self._handle_run_selected_tests,
        )

        # 8. apply_patch
        self.register_tool(
            ToolDefinition(
                name="apply_patch",
                description="Apply a verified unified diff patch to the target repository workspace. Requires developer approval.",
                parameters={
                    "type": "object",
                    "properties": {
                        "unified_diff": {"type": "string", "description": "Standard git-compatible unified diff string to apply."},
                    },
                    "required": ["unified_diff"],
                },
                tool_type=ToolType.CONSEQUENTIAL,
                requires_authorization=True,
            ),
            self._handle_apply_patch,
        )

    # Handlers

    def _handle_search_codebase(self, args: Dict[str, Any]) -> str:
        query = args.get("query", "")
        top_k = int(args.get("top_k", 5))

        if not self.retriever:
            return "Search unavailable: HybridRetriever is not configured."

        results = self.retriever.search(query, top_k=top_k, mode="hybrid")
        if not results:
            return f"No code chunks matched query: '{query}'"

        formatted = []
        for i, res in enumerate(results, start=1):
            chunk = res.chunk
            sym = chunk.symbol_name or chunk.parent_symbol or "module"
            formatted.append(
                f"[{i}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line} ({chunk.chunk_type.value} {sym}):\n"
                f"{chunk.content}\n"
            )
        return "\n".join(formatted)

    def _handle_read_source_file(self, args: Dict[str, Any]) -> str:
        file_path = args.get("file_path", "")
        target = self.policy_engine.validate_file_access(file_path)

        if not target.is_file():
            return f"File not found: '{file_path}'"

        content = target.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()

        start_line = args.get("start_line")
        end_line = args.get("end_line")

        if start_line is not None or end_line is not None:
            s = max(1, int(start_line)) if start_line is not None else 1
            e = min(len(lines), int(end_line)) if end_line is not None else len(lines)
            selected = lines[s - 1:e]
            numbered = [f"{s + idx}: {line}" for idx, line in enumerate(selected)]
            return f"--- File: {file_path} (lines {s}-{e} of {len(lines)}) ---\n" + "\n".join(numbered)

        # Full file up to 200 lines
        max_lines = 200
        if len(lines) > max_lines:
            truncated = lines[:max_lines]
            numbered = [f"{idx + 1}: {line}" for idx, line in enumerate(truncated)]
            return (
                f"--- File: {file_path} (first {max_lines} of {len(lines)} lines) ---\n"
                + "\n".join(numbered)
                + f"\n... [Truncated. Use start_line/end_line to inspect remaining {len(lines) - max_lines} lines]"
            )

        numbered = [f"{idx + 1}: {line}" for idx, line in enumerate(lines)]
        return f"--- File: {file_path} ({len(lines)} lines) ---\n" + "\n".join(numbered)

    def _handle_find_related_tests(self, args: Dict[str, Any]) -> str:
        query = args.get("symbol_or_path", "")
        clean_name = Path(query).stem.replace("test_", "")

        # Look for test files in repo
        test_files = []
        for p in self.repository_path.rglob("*.py"):
            if "test" in p.name.lower() or "tests" in p.parts:
                if clean_name.lower() in p.name.lower() or not query:
                    try:
                        rel = p.relative_to(self.repository_path).as_posix()
                        test_files.append(rel)
                    except ValueError:
                        pass

        if not test_files:
            return f"No existing test files found matching '{query}'."

        return f"Found {len(test_files)} related test file(s):\n" + "\n".join(f"  • {f}" for f in test_files[:10])

    def _handle_inspect_dependencies(self, args: Dict[str, Any]) -> str:
        manifest_names = ["pyproject.toml", "requirements.txt", "setup.py", "Pipfile", "environment.yml"]
        found_manifests = []

        for name in manifest_names:
            try:
                validated_path = self.policy_engine.validate_file_access(name)
            except SecurityPolicyError:
                logger.warning("Manifest '%s' violates workspace boundary or escapes via symlink; skipping.", name)
                continue

            if validated_path.is_file():
                try:
                    content = validated_path.read_text(encoding="utf-8", errors="replace")
                    found_manifests.append(f"--- {name} ---\n{content[:2000]}")
                except Exception as e:
                    found_manifests.append(f"--- {name} (Error reading: {e}) ---")

        if not found_manifests:
            return "No dependency manifests found in repository root."

        return "\n\n".join(found_manifests)

    def _handle_generate_code_proposal(self, args: Dict[str, Any]) -> str:
        task = args.get("task", "")
        target_file = args.get("target_file")

        if not self.code_generator:
            return "Code proposal generator is not configured."

        result = self.code_generator.propose(task=task, target_file=target_file)
        self.collected_proposals.append(result)

        if not result.proposals:
            return f"Failed to generate code proposal: {result.verification_summary}"

        prop = result.proposals[0]
        ast_status = "Valid Python" if prop.ast_valid else f"AST Error: {prop.syntax_errors}"
        return (
            f"Code proposal generated for '{prop.target_file}':\n"
            f"AST Verification: {ast_status}\n\n"
            f"--- Architectural Reasoning ---\n{prop.architectural_reasoning}\n\n"
            f"--- Unified Diff ---\n{prop.unified_diff}"
        )

    def _handle_generate_test_proposal(self, args: Dict[str, Any]) -> str:
        source_file = args.get("source_file", "")
        framework = args.get("framework")

        if not self.test_generator:
            return "Test proposal generator is not configured."

        result = self.test_generator.propose_tests(source_file=source_file, framework=framework)
        self.collected_proposals.append(result)

        if not result.proposals:
            return f"Failed to generate test proposal: {result.verification_summary}"

        prop = result.proposals[0]
        ast_status = "Valid Python" if prop.ast_valid else f"AST Error: {prop.syntax_errors}"
        return (
            f"Test proposal generated for '{prop.target_file}':\n"
            f"AST Verification: {ast_status}\n\n"
            f"--- Architectural Reasoning ---\n{prop.architectural_reasoning}\n\n"
            f"--- Unified Diff ---\n{prop.unified_diff}"
        )

    def _handle_run_selected_tests(self, args: Dict[str, Any]) -> str:
        command = args.get("command", "")
        cmd_args = self.policy_engine.validate_test_command(command)

        auth_req = AuthorizationRequest(
            tool_name="run_selected_tests",
            arguments=args,
            action_summary=f"Execute test runner command: {command}",
            risk_assessment="Executes local subprocess in isolated disposable workspace without network access.",
        )

        authorized = self.permission_engine.check_authorization(ToolType.CONSEQUENTIAL, auth_req)
        if not authorized:
            raise PermissionError(f"User denied authorization for test execution: '{command}'.")

        # Network isolation: active loopback proxies block outbound connections
        clean_env = os.environ.copy()
        clean_env["HTTP_PROXY"] = "http://127.0.0.1:0"
        clean_env["HTTPS_PROXY"] = "http://127.0.0.1:0"
        clean_env["ALL_PROXY"] = "socks5://127.0.0.1:0"
        clean_env["http_proxy"] = "http://127.0.0.1:0"
        clean_env["https_proxy"] = "http://127.0.0.1:0"
        clean_env["all_proxy"] = "socks5://127.0.0.1:0"
        clean_env["NO_PROXY"] = ""
        clean_env["no_proxy"] = ""

        # Execute tests in a disposable temporary workspace copy
        with tempfile.TemporaryDirectory(prefix="agent_test_sandbox_") as tmp_dir:
            sandbox_repo = Path(tmp_dir) / "repo"

            # Filter unsafe symlinks and heavy cache/VCS directories during copy
            def _ignore_filter(directory: str, contents: List[str]) -> set[str]:
                ignored: set[str] = set()
                dir_path = Path(directory)
                for name in contents:
                    if name in (".git", "__pycache__", ".pytest_cache", ".venv", "venv", ".mypy_cache"):
                        ignored.add(name)
                    elif name.endswith(".pyc"):
                        ignored.add(name)
                    else:
                        item_path = dir_path / name
                        if item_path.is_symlink():
                            try:
                                resolved = item_path.resolve()
                                resolved.relative_to(self.repository_path)
                            except (ValueError, RuntimeError):
                                ignored.add(name)
                return ignored

            shutil.copytree(self.repository_path, sandbox_repo, ignore=_ignore_filter, symlinks=False)

            # Map any arguments referencing the repository path into the disposable sandbox
            remapped_cmd: List[str] = []
            for arg in cmd_args:
                if arg.startswith("-") and "=" in arg:
                    opt, val = arg.split("=", 1)
                    val_path = Path(val)
                    if val_path.is_absolute():
                        try:
                            rel = val_path.relative_to(self.repository_path)
                            remapped_cmd.append(f"{opt}={sandbox_repo / rel}")
                            continue
                        except ValueError:
                            pass
                    remapped_cmd.append(arg)
                else:
                    arg_path = Path(arg)
                    if arg_path.is_absolute():
                        try:
                            rel = arg_path.relative_to(self.repository_path)
                            remapped_cmd.append(str(sandbox_repo / rel))
                            continue
                        except ValueError:
                            pass
                    remapped_cmd.append(arg)

            existing_pythonpath = clean_env.get("PYTHONPATH", "")
            filtered_pp = [
                entry for entry in existing_pythonpath.split(os.pathsep)
                if entry and not entry.startswith(str(self.repository_path))
            ]
            clean_env["PYTHONPATH"] = os.pathsep.join(
                [str(sandbox_repo / "src"), str(sandbox_repo)] + filtered_pp
            )

            try:
                res = subprocess.run(
                    remapped_cmd,
                    cwd=sandbox_repo,
                    capture_output=True,
                    text=True,
                    timeout=self.command_timeout_seconds,
                    env=clean_env,
                )
                out = res.stdout.strip()
                err = res.stderr.strip()
                status = "PASSED" if res.returncode == 0 else f"FAILED (Exit Code {res.returncode})"
                return (
                    f"Test Execution Result: {status}\n"
                    f"--- STDOUT ---\n{out or '(empty)'}\n"
                    f"--- STDERR ---\n{err or '(empty)'}"
                )
            except subprocess.TimeoutExpired:
                return f"Test Execution Timed Out after {self.command_timeout_seconds} seconds."

    @staticmethod
    def _split_unified_diff(unified_diff: str) -> List[Tuple[str, str]]:
        """Split a potentially multi-file unified diff into individual (target_path, file_diff) pairs."""
        lines = unified_diff.splitlines(keepends=True)
        chunks: List[List[str]] = []
        current_chunk: List[str] = []

        for line in lines:
            # Detect boundary between files:
            # Either 'diff --git ' or a '--- ' header when current_chunk already contains '+++ '
            if line.startswith("diff --git ") or (
                line.startswith("--- ") and any(l.startswith("+++ ") for l in current_chunk)
            ):
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
            current_chunk.append(line)

        if current_chunk:
            chunks.append(current_chunk)

        results: List[Tuple[str, str]] = []
        for chunk in chunks:
            target_path: Optional[str] = None
            for line in chunk:
                if line.startswith("+++ "):
                    raw = line[4:].strip()
                    clean = raw[2:] if raw.startswith("b/") else raw
                    if clean and clean != "/dev/null":
                        target_path = clean
                        break
                elif line.startswith("--- ") and target_path is None:
                    raw = line[4:].strip()
                    clean = raw[2:] if raw.startswith("a/") else raw
                    if clean and clean != "/dev/null":
                        target_path = clean

            if target_path:
                results.append((target_path, "".join(chunk)))

        return results

    def _handle_apply_patch(self, args: Dict[str, Any]) -> str:
        unified_diff = args.get("unified_diff", "")
        target_paths = self.policy_engine.validate_patch_targets(unified_diff)

        target_desc = ", ".join(p.name for p in target_paths)
        auth_req = AuthorizationRequest(
            tool_name="apply_patch",
            arguments=args,
            action_summary=f"Apply unified diff patch to: {target_desc}",
            risk_assessment="Modifies target files in the repository workspace.",
            target_path=str(target_paths[0]) if len(target_paths) == 1 else None,
        )

        authorized = self.permission_engine.check_authorization(ToolType.CONSEQUENTIAL, auth_req)
        if not authorized:
            raise PermissionError("User denied authorization for applying patch.")

        file_diffs = self._split_unified_diff(unified_diff)
        if not file_diffs:
            raise ValueError("No valid file diffs found in patch payload.")

        # Phase 1: In-memory dry run and AST verification for all files (no disk writes)
        prepared_updates: List[Tuple[str, Path, str, str, bool]] = []
        for rel_file, file_diff in file_diffs:
            abs_target = self.policy_engine.validate_file_access(rel_file)
            existed = abs_target.is_file()
            orig_content = (
                abs_target.read_text(encoding="utf-8", errors="replace")
                if existed
                else ""
            )

            # Dry-run patch in memory
            try:
                patched_content = DiffEngine.apply_diff_in_memory(orig_content, file_diff)
            except Exception as e:
                raise ValueError(f"Failed to dry-run patch in memory for '{rel_file}': {e}") from e

            # AST verify Python files deterministically
            if abs_target.suffix == ".py":
                is_valid, errors = DiffEngine.validate_ast_syntax(patched_content, filename=rel_file)
                if not is_valid:
                    raise ValueError(f"Patched code for '{rel_file}' failed AST validation: {'; '.join(errors)}")

            prepared_updates.append((rel_file, abs_target, orig_content, patched_content, existed))

        # Phase 2: Atomic disk application with rollback guarantee on write failure
        written_stack: List[Tuple[Path, str, bool]] = []
        modified_files: List[str] = []

        try:
            for rel_file, abs_target, orig_content, patched_content, existed in prepared_updates:
                written_stack.append((abs_target, orig_content, existed))
                abs_target.parent.mkdir(parents=True, exist_ok=True)
                abs_target.write_text(patched_content, encoding="utf-8")
                modified_files.append(rel_file)
        except Exception as write_err:
            # Rollback modified files to original contents in reverse order
            for target, orig, existed in reversed(written_stack):
                try:
                    if existed:
                        target.write_text(orig, encoding="utf-8")
                    else:
                        if target.exists():
                            target.unlink()
                except Exception as rb_err:
                    logger.error("Failed to rollback file '%s': %s", target, rb_err)
            raise RuntimeError(
                f"Atomic patch application failed during disk write. Rolled back all changes: {write_err}"
            ) from write_err

        return f"Successfully applied patch to {len(modified_files)} file(s): {', '.join(modified_files)}"
