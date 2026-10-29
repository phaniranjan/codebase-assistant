"""Tool definitions and registry for Agent execution."""

import json
import logging
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional

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
            p = self.repository_path / name
            if p.is_file():
                try:
                    content = p.read_text(encoding="utf-8", errors="replace")
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
            risk_assessment="Executes local subprocess in isolated workspace without network access.",
        )

        authorized = self.permission_engine.check_authorization(ToolType.CONSEQUENTIAL, auth_req)
        if not authorized:
            raise PermissionError(f"User denied authorization for test execution: '{command}'.")

        # Strip network proxies from execution environment
        clean_env = os.environ.copy()
        for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"]:
            clean_env.pop(k, None)

        try:
            res = subprocess.run(
                cmd_args,
                cwd=self.repository_path,
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

        # In-memory dry run and AST verification before writing
        diff_lines = unified_diff.splitlines()
        current_file = None
        per_file_diffs: Dict[str, List[str]] = {}

        for line in diff_lines:
            if line.startswith("+++ "):
                raw_path = line[4:].strip()
                clean_path = raw_path[2:] if raw_path.startswith("b/") else raw_path
                current_file = clean_path
                per_file_diffs[current_file] = []
            if current_file is not None:
                per_file_diffs[current_file].append(line)

        modified_files = []
        for rel_file, d_lines in per_file_diffs.items():
            if not rel_file or rel_file == "/dev/null":
                continue
            abs_target = self.policy_engine.validate_file_access(rel_file)
            file_diff = "\n".join(d_lines)

            orig_content = (
                abs_target.read_text(encoding="utf-8", errors="replace")
                if abs_target.is_file()
                else ""
            )

            # Dry-run patch in memory
            patched_content = DiffEngine.apply_diff_in_memory(orig_content, file_diff)

            # AST verify Python files
            if abs_target.suffix == ".py":
                is_valid, errors = DiffEngine.validate_ast_syntax(patched_content, filename=rel_file)
                if not is_valid:
                    raise ValueError(f"Patched code for '{rel_file}' failed AST validation: {'; '.join(errors)}")

            # Apply to disk
            abs_target.parent.mkdir(parents=True, exist_ok=True)
            abs_target.write_text(patched_content, encoding="utf-8")
            modified_files.append(rel_file)

        return f"Successfully applied patch to {len(modified_files)} file(s): {', '.join(modified_files)}"
