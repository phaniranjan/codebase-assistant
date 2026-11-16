"""Model Context Protocol (MCP) server for Google Antigravity conversational IDE integration."""

import json
import logging
import os
import re
import secrets
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Compatibility import for official MCP Python SDK:
# Supports mcp 1.x (FastMCP) and mcp 2.x (MCPServer)
try:
    from mcp.server.fastmcp import FastMCP  # type: ignore[attr-defined]
except (ImportError, ModuleNotFoundError):
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:
        FastMCP = None  # type: ignore

from codebase_assistant.agent.coordinator import AgentCoordinator
from codebase_assistant.agent.diagnostics import DiagnosticEngine, sanitize_untrusted_text
from codebase_assistant.agent.models import AgentConfig
from codebase_assistant.agent.permissions import PermissionEngine
from codebase_assistant.agent.policy import PolicyEngine, SecurityPolicyError
from codebase_assistant.agent.repair import RepairWorkflowCoordinator
from codebase_assistant.agent.repair_models import RepairWorkflowStatus
from codebase_assistant.agent.tools import ToolRegistry
from codebase_assistant.config import get_settings
from codebase_assistant.embeddings.gemini import GeminiEmbeddingProvider
from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.embeddings.vector_store import VectorStore
from codebase_assistant.generation.code_generator import CodeProposalGenerator
from codebase_assistant.generation.diff_engine import DiffEngine
from codebase_assistant.generation.test_generator import TestProposalGenerator
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.llm.client import GeminiClient
from codebase_assistant.llm.mock_client import MockGeminiClient
from codebase_assistant.llm.qa import CodebaseQAEngine
from codebase_assistant.retrieval.hybrid import HybridRetriever

# Standardize logger to write exclusively to stderr so JSON-RPC on stdout is never polluted
logger = logging.getLogger("codebase_assistant.mcp_server")


class MCPBackend:
    """Encapsulates repository state, caching, retrieval, and agent service instances for MCP."""

    PROTECTED_REPOS = {"ai-evaluation", "ai_evaluation"}

    def __init__(self, is_live: Optional[bool] = None, allow_fallback: bool = False) -> None:
        self.settings = get_settings()
        self.allow_fallback = allow_fallback
        self.is_live = (
            is_live
            if is_live is not None
            else bool(self.settings.gemini_api_key or os.environ.get("GEMINI_API_KEY"))
        )

        self.active_repo: Optional[Path] = None
        self.cache_dir: Optional[Path] = None
        self.is_active_repo_writable: bool = False

        # Core engine components
        self.indexer: Optional[IncrementalIndexer] = None
        self.vector_store: Optional[VectorStore] = None
        self.retriever: Optional[HybridRetriever] = None
        self.qa_engine: Optional[CodebaseQAEngine] = None
        self.tool_registry: Optional[ToolRegistry] = None
        self.policy_engine: Optional[PolicyEngine] = None
        self.code_generator: Optional[CodeProposalGenerator] = None
        self.test_generator: Optional[TestProposalGenerator] = None
        self.agent_coordinator: Optional[AgentCoordinator] = None
        self.repair_coordinator: Optional[RepairWorkflowCoordinator] = None

        # Providers
        self.emb_provider: Any = None
        self.llm_client: Any = None
        self._init_providers()

        # One-time server authorization tokens: token -> (canonical_workspace, expires_at)
        self._pending_tokens: Dict[str, Tuple[Path, float]] = {}
        self.writable_workspaces: Set[Path] = set()

    def _init_providers(self) -> None:
        """Initialize embedding and LLM providers depending on live vs offline mode."""
        if self.is_live:
            try:
                self.emb_provider = GeminiEmbeddingProvider()
                self.llm_client = GeminiClient()
                logger.info("MCP server initialized with live Google Gemini providers.")
                return
            except Exception as e:
                if not self.allow_fallback:
                    raise RuntimeError(
                        f"Failed to initialize live Gemini providers ({e}). "
                        "Live mode requires valid credentials. Pass allow_fallback=True or --mock to use offline mode."
                    ) from e
                logger.warning(
                    "Failed to initialize live Gemini providers (%s). Falling back to offline mock mode (--allow-fallback is enabled).",
                    e,
                )
                self.is_live = False

        self.emb_provider = MockEmbeddingProvider()
        self.llm_client = MockGeminiClient()
        logger.info("MCP server initialized with offline deterministic mock providers.")

    def is_protected_repository(self, path: Path | str) -> bool:
        """Check if a path points to or resides within a protected read-only repository."""
        canonical = Path(path).resolve()
        real = Path(os.path.realpath(canonical))
        for check_p in (canonical, real):
            if check_p.name in self.PROTECTED_REPOS:
                return True
            for part in check_p.parts:
                if part in self.PROTECTED_REPOS:
                    return True
        return False

    def is_workspace_authorized(self, target_dir: Path | str) -> bool:
        """Verify that a workspace is registered, canonical, and not a protected repository."""
        canonical = Path(target_dir).resolve()
        real = Path(os.path.realpath(canonical))
        if canonical != real:
            logger.warning(
                "Workspace path resolves to a different realpath: %s != %s", canonical, real
            )
            return False
        if not canonical.is_dir():
            return False
        if self.is_protected_repository(canonical):
            return False
        return canonical in self.writable_workspaces

    def register_writable_workspace(self, workspace_path: Path | str) -> Path:
        """Explicitly register a disposable development workspace as writable."""
        canonical = Path(workspace_path).resolve()
        real = Path(os.path.realpath(canonical))
        if canonical != real:
            raise ValueError(f"Symlink or path mismatch for workspace: {canonical} != {real}")
        if not canonical.is_dir():
            raise NotADirectoryError(f"Workspace path is not a directory: {canonical}")
        if self.is_protected_repository(canonical):
            raise PermissionError(
                f"Cannot register protected repository '{canonical}' as writable."
            )
        self.writable_workspaces.add(canonical)
        return canonical

    def unregister_writable_workspace(self, workspace_path: Path | str) -> None:
        """Remove a workspace from the authorized writable set."""
        canonical = Path(workspace_path).resolve()
        self.writable_workspaces.discard(canonical)

    def issue_authorization_token(self, workspace_path: Path | str) -> str:
        """Issue a cryptographically secure, one-time authorization token for a registered workspace."""
        canonical = Path(workspace_path).resolve()
        if not self.is_workspace_authorized(canonical):
            raise PermissionError(
                f"Cannot issue authorization token for unauthorized workspace: {canonical}"
            )
        token = secrets.token_hex(32)
        self._pending_tokens[token] = (canonical, time.time() + 60.0)
        return token

    def verify_and_consume_token(self, token: Optional[str], workspace_path: Path | str) -> bool:
        """Verify and consume a one-time authorization token."""
        if not token:
            return False
        entry = self._pending_tokens.pop(token, None)
        if not entry:
            return False
        target_ws, expires_at = entry
        if time.time() > expires_at:
            return False
        canonical = Path(workspace_path).resolve()
        return canonical == target_ws

    def set_active_repository(
        self,
        repository_path: str,
        cache_dir: Optional[str] = None,
        writable: bool = False,
    ) -> Dict[str, Any]:
        """Establish the active repository and initialize indexing and retrieval components."""
        target_path = Path(repository_path).resolve()
        real_target = Path(os.path.realpath(target_path))
        if target_path != real_target:
            raise ValueError(
                f"Symlink escape or path mismatch detected: {target_path} != {real_target}"
            )
        if not target_path.exists():
            raise FileNotFoundError(f"Target repository path does not exist: '{target_path}'")
        if not target_path.is_dir():
            raise NotADirectoryError(f"Target repository path is not a directory: '{target_path}'")

        # Protected repository enforcement: protected repos can NEVER be writable
        if self.is_protected_repository(target_path):
            writable = False

        # Establish cache directory strictly outside the target repository for read-only safety
        if cache_dir:
            resolved_cache = Path(cache_dir).resolve()
            try:
                resolved_cache.relative_to(target_path)
                raise ValueError(
                    f"Read-only invariant violation: Cache directory '{resolved_cache}' is located inside the target repository '{target_path}'."
                )
            except ValueError as e:
                if "Read-only invariant violation" in str(e):
                    raise
                self.cache_dir = resolved_cache
        else:
            repo_hash = secrets.token_hex(4)
            safe_name = f"{target_path.name}_{repo_hash}"
            self.cache_dir = (
                Path(tempfile.gettempdir()).resolve() / "codebase_assistant_mcp" / safe_name
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.active_repo = target_path
        self.is_active_repo_writable = writable
        if writable:
            self.writable_workspaces.add(target_path)
        else:
            self.writable_workspaces.discard(target_path)

        # 1. Run incremental indexing
        self.indexer = IncrementalIndexer(self.active_repo, cache_dir=self.cache_dir)
        manifest = self.indexer.load_manifest()
        if manifest is None:
            self.indexer.index(rebuild=True)
        chunks = self.indexer.load_chunks()
        if not chunks:
            self.indexer.index()
            chunks = self.indexer.load_chunks()

        # 2. Synchronize embeddings & vector store
        manager = EmbeddingManager(
            self.active_repo, cache_dir=self.cache_dir, provider=self.emb_provider
        )
        self.vector_store = manager.load_vector_store()
        if self.vector_store.count() == 0:
            manager.sync(rebuild=True)
            self.vector_store = manager.load_vector_store()

        # 3. Initialize hybrid retriever
        self.retriever = HybridRetriever(
            chunks=chunks,
            vector_store=self.vector_store,
            embedding_provider=self.emb_provider,
        )

        # 4. Initialize QA engine
        self.qa_engine = CodebaseQAEngine(
            retriever=self.retriever,
            llm_client=self.llm_client,
            default_budget=8192,
        )

        # 5. Initialize proposal generators & policy engine
        self.policy_engine = PolicyEngine(self.active_repo)
        self.code_generator = CodeProposalGenerator(
            repository_path=self.active_repo,
            retriever=self.retriever,
            llm_client=self.llm_client,
        )
        self.test_generator = TestProposalGenerator(
            repository_path=self.active_repo,
            retriever=self.retriever,
            llm_client=self.llm_client,
        )

        # 6. Initialize ToolRegistry & Agent Coordinator
        self.tool_registry = ToolRegistry(
            repository_path=self.active_repo,
            retriever=self.retriever,
            code_generator=self.code_generator,
            test_generator=self.test_generator,
            policy_engine=self.policy_engine,
        )
        self.agent_coordinator = AgentCoordinator(
            tool_registry=self.tool_registry,
            llm_client=self.llm_client,
            config=AgentConfig(max_iterations=5, max_reasoning_steps=10, token_budget=8192),
        )

        # 7. Initialize Repair Workflow Coordinator (disposable sandbox execution)
        repair_perm_engine = PermissionEngine(auto_approve_consequential=True)
        self.repair_coordinator = RepairWorkflowCoordinator(
            repository_path=self.active_repo,
            code_generator=self.code_generator,
            test_generator=self.test_generator,
            retriever=self.retriever,
            gemini_client=self.llm_client,
            policy_engine=self.policy_engine,
            permission_engine=repair_perm_engine,
            max_retries=3,
        )

        logger.info(
            "Active repository set to '%s' (chunks: %d, cache: '%s', writable: %s)",
            self.active_repo,
            len(chunks),
            self.cache_dir,
            self.is_active_repo_writable,
        )

        return {
            "status": "ready",
            "repository_path": str(self.active_repo),
            "repository_name": self.active_repo.name,
            "total_chunks": len(chunks),
            "total_vectors": self.vector_store.count(),
            "cache_directory": str(self.cache_dir),
            "execution_mode": "LIVE GEMINI" if self.is_live else "OFFLINE MOCK",
            "is_writable": self.is_active_repo_writable,
        }

    def require_active_repo(self) -> Path:
        """Ensure an active repository is set before tool execution."""
        if self.active_repo is None or self.retriever is None:
            raise RuntimeError(
                "No active repository selected. Please call 'set_active_repository(repository_path=...)' first."
            )
        return self.active_repo

    def get_status(self) -> Dict[str, Any]:
        """Return the current repository identity and status."""
        if self.active_repo is None or self.retriever is None:
            return {
                "status": "uninitialized",
                "message": "No active repository selected. Call 'set_active_repository(repository_path=...)' to begin.",
                "execution_mode": "LIVE GEMINI" if self.is_live else "OFFLINE MOCK",
                "capabilities": [
                    "search_codebase",
                    "read_source_file",
                    "ask_codebase",
                    "propose_code",
                    "propose_tests",
                    "inspect_dependencies",
                    "investigate_task",
                    "repair_code",
                    "apply_approved_patch",
                ],
            }

        return {
            "status": "active",
            "repository_path": str(self.active_repo),
            "repository_name": self.active_repo.name,
            "total_chunks": len(self.retriever.chunks),
            "total_vectors": self.vector_store.count() if self.vector_store else 0,
            "cache_directory": str(self.cache_dir),
            "execution_mode": "LIVE GEMINI" if self.is_live else "OFFLINE MOCK",
            "model_name": getattr(self.llm_client, "model_name", "unknown"),
            "is_writable": self.is_active_repo_writable,
            "capabilities": [
                "search_codebase",
                "read_source_file",
                "ask_codebase",
                "propose_code",
                "propose_tests",
                "inspect_dependencies",
                "investigate_task",
                "repair_code",
                "apply_approved_patch",
            ],
        }


def create_mcp_server(backend: Optional[MCPBackend] = None) -> Any:
    """Construct and configure the FastMCP/MCPServer instance with registered tools."""
    if FastMCP is None:
        raise ImportError(
            "The official 'mcp' SDK is required for Antigravity MCP integration. Install via: pip install 'mcp>=1.0.0'"
        )

    server = FastMCP(
        name="codebase-assistant",
        instructions=(
            "Codebase Assistant MCP Server: Conversational interface for code exploration, "
            "grounded question answering with line citations, unified diff proposals, and sandboxed self-repair."
        ),
    )
    be = backend or MCPBackend()

    # -------------------------------------------------------------------------
    # Tool 1: set_active_repository
    # -------------------------------------------------------------------------
    @server.tool()
    def set_active_repository(
        repository_path: str,
        cache_dir: Optional[str] = None,
        writable: bool = False,
    ) -> str:
        """Select and initialize the active target repository for analysis and code proposals.

        Args:
            repository_path: Absolute or workspace path to the repository directory.
            cache_dir: Optional external cache directory path (must be outside the repository).
            writable: Set True ONLY if this repository checkout is an explicitly writable dev workspace.
        """
        try:
            res = be.set_active_repository(
                repository_path=repository_path, cache_dir=cache_dir, writable=writable
            )
            mode_str = res["execution_mode"]
            return (
                f"### Repository Activated: `{res['repository_name']}`\n"
                f"- **Path**: `{res['repository_path']}`\n"
                f"- **Indexed Chunks**: {res['total_chunks']}\n"
                f"- **Vectors**: {res['total_vectors']}\n"
                f"- **Execution Mode**: {mode_str}\n"
                f"- **Writable Mode**: {'Yes (Modifications Allowed)' if res['is_writable'] else 'No (Read-Only Knowledge Source)'}\n"
                f"- **Cache Location**: `{res['cache_directory']}`\n\n"
                f"You can now ask questions, search symbols, generate diff proposals, or run repair workflows."
            )
        except Exception as e:
            logger.exception("Error in set_active_repository: %s", e)
            return f"Error setting active repository: {e}"

    # -------------------------------------------------------------------------
    # Tool 2: repository_status
    # -------------------------------------------------------------------------
    @server.tool()
    def repository_status() -> str:
        """Get the current active repository status, index statistics, and capabilities."""
        try:
            status = be.get_status()
            if status["status"] == "uninitialized":
                return (
                    "### Codebase Assistant: No Repository Active\n"
                    "No repository is currently selected. Use `set_active_repository(repository_path=...)` "
                    "to select and index a target repository."
                )

            return (
                f"### Codebase Assistant Status: Active\n"
                f"- **Repository**: `{status['repository_name']}` (`{status['repository_path']}`)\n"
                f"- **Chunks**: {status['total_chunks']} indexed chunks\n"
                f"- **Vectors**: {status['total_vectors']} vectors synchronized\n"
                f"- **Mode**: {status['execution_mode']} (Model: `{status['model_name']}`)\n"
                f"- **Writable Workspace**: {'Yes' if status['is_writable'] else 'No (Protected Read-Only)'}\n"
                f"- **Available Tools**: {', '.join(status['capabilities'])}"
            )
        except Exception as e:
            logger.exception("Error in repository_status: %s", e)
            return f"Error getting repository status: {e}"

    # -------------------------------------------------------------------------
    # Tool 3: search_codebase
    # -------------------------------------------------------------------------
    @server.tool()
    def search_codebase(
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> str:
        """Search the indexed codebase for code symbols, functions, classes, and architectural concepts.

        Args:
            query: Natural language or symbol query (e.g., 'authentication token validator' or 'RAGPipeline').
            top_k: Maximum number of relevant code chunks to return (default: 5).
            mode: Search mode: 'hybrid' (vector + BM25 with RRF), 'dense_only', or 'sparse_only'.
        """
        try:
            be.require_active_repo()
            assert be.retriever is not None
            clean_mode = mode.lower()
            if clean_mode not in ("hybrid", "dense_only", "sparse_only"):
                clean_mode = "hybrid"

            results = be.retriever.search(query=query, top_k=top_k, mode=clean_mode)  # type: ignore
            if not results:
                return f"No code chunks matched the query: '{query}'."

            out_lines = [
                f"### Codebase Search Results for: *{query}* (Mode: {clean_mode.upper()})\n"
            ]
            for idx, res in enumerate(results, start=1):
                sym = res.chunk.symbol_name or res.chunk.chunk_type.value
                out_lines.append(
                    f"**{idx}. `{res.chunk.file_path}`** (Lines {res.chunk.start_line}–{res.chunk.end_line}) "
                    f"[{res.chunk.chunk_type.value}: `{sym}`] — Score: {res.score:.4f}\n"
                    f"```python\n{res.chunk.content.strip()}\n```\n"
                )
            return "\n".join(out_lines)
        except Exception as e:
            logger.exception("Error in search_codebase: %s", e)
            return f"Error searching codebase: {e}"

    # -------------------------------------------------------------------------
    # Tool 4: read_source_file
    # -------------------------------------------------------------------------
    @server.tool()
    def read_source_file(
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Safely read source file contents with line numbers from the active repository.

        Args:
            file_path: Relative path to the file within the repository (e.g., 'src/api/client.py').
            start_line: Optional 1-based start line number.
            end_line: Optional 1-based end line number.
        """
        try:
            repo = be.require_active_repo()
            target_path = DiffEngine.validate_target_path(repo, file_path)
            if not target_path.is_file():
                return f"Error: File does not exist: '{file_path}'"

            content = target_path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            s_line = max(1, start_line) if start_line is not None else 1
            e_line = min(len(lines), end_line) if end_line is not None else len(lines)

            if s_line > len(lines):
                return f"Error: start_line ({s_line}) exceeds total line count ({len(lines)}) of '{file_path}'."

            selected_lines = lines[s_line - 1 : e_line]
            formatted_lines = [
                f"{line_num:4d} | {line}"
                for line_num, line in enumerate(selected_lines, start=s_line)
            ]

            return (
                f"### File: `{file_path}` (Lines {s_line}–{e_line} of {len(lines)})\n"
                f"```python\n" + "\n".join(formatted_lines) + "\n```"
            )
        except Exception as e:
            logger.exception("Error in read_source_file: %s", e)
            return f"Error reading source file '{file_path}': {e}"

    # -------------------------------------------------------------------------
    # Tool 5: ask_codebase
    # -------------------------------------------------------------------------
    @server.tool()
    def ask_codebase(
        question: str,
        top_k: int = 5,
        mode: str = "hybrid",
        budget: int = 8192,
    ) -> str:
        """Ask technical questions about the codebase and receive grounded answers with strict line citations.

        Args:
            question: Developer question about architecture, data flow, error handling, or symbols.
            top_k: Number of context chunks to assemble.
            mode: Retrieval mode ('hybrid', 'dense_only', 'sparse_only').
            budget: Maximum token budget for context assembly.
        """
        try:
            be.require_active_repo()
            assert be.qa_engine is not None
            clean_mode = mode.lower()
            if clean_mode not in ("hybrid", "dense_only", "sparse_only"):
                clean_mode = "hybrid"

            resp = be.qa_engine.ask(
                query=question,
                top_k=top_k,
                mode=clean_mode,  # type: ignore
                budget=budget,
            )

            citations_md = ""
            if resp.citations:
                cit_items = [
                    f"- `{c.file_path}` (Lines {c.start_line}–{c.end_line})" for c in resp.citations
                ]
                citations_md = "\n\n**Verified Citations:**\n" + "\n".join(cit_items)

            return f"### Grounded Response\n\n{resp.answer}{citations_md}"
        except Exception as e:
            logger.exception("Error in ask_codebase: %s", e)
            return f"Error answering question: {e}"

    # -------------------------------------------------------------------------
    # Tool 6: propose_code
    # -------------------------------------------------------------------------
    @server.tool()
    def propose_code(
        task: str,
        target_file: str,
        top_k: int = 5,
        budget: int = 8192,
    ) -> str:
        """Generate a context-aware unified diff code proposal with AST validation and architectural reasoning.

        Args:
            task: Clear description of the code modification or feature to implement.
            target_file: Relative path to the file to modify or create (e.g., 'src/models.py').
            top_k: Number of relevant code chunks to retrieve for grounding.
            budget: Token budget for context assembly.
        """
        try:
            repo = be.require_active_repo()
            assert be.code_generator is not None
            DiffEngine.validate_target_path(repo, target_file)
            proposal = be.code_generator.propose(
                task=task,
                target_file=target_file,
                top_k=top_k,
                budget=budget,
            )

            validation_badge = "PASSED" if proposal.is_valid else "FAILED"
            first_p = proposal.proposals[0] if proposal.proposals else None
            diff_text = first_p.unified_diff if first_p else ""
            reasoning = (
                first_p.architectural_reasoning if first_p else proposal.verification_summary
            )
            return (
                f"### Code Proposal for `{target_file}`\n"
                f"- **Task**: {task}\n"
                f"- **AST Validation**: `{validation_badge}`\n\n"
                f"#### Architectural Reasoning\n{reasoning}\n\n"
                f"#### Proposed Unified Diff\n"
                f"```diff\n{diff_text}\n```\n\n"
                f"*To apply this diff, review the changes and run `apply_approved_patch`.*"
            )
        except Exception as e:
            logger.exception("Error in propose_code: %s", e)
            return f"Error generating code proposal: {e}"

    # -------------------------------------------------------------------------
    # Tool 7: propose_tests
    # -------------------------------------------------------------------------
    @server.tool()
    def propose_tests(
        task: str,
        target_file: str,
        top_k: int = 5,
        budget: int = 8192,
    ) -> str:
        """Generate targeted unit tests matching the repository's test framework (pytest vs unittest).

        Args:
            task: Description of what behaviors, edge cases, or regressions the tests should cover.
            target_file: Relative path to the test file to create or update (e.g., 'tests/unit/test_api.py').
            top_k: Number of relevant code chunks to retrieve for grounding.
            budget: Token budget for context assembly.
        """
        try:
            repo = be.require_active_repo()
            assert be.test_generator is not None
            DiffEngine.validate_target_path(repo, target_file)
            proposal = be.test_generator.propose_tests(
                source_file=target_file,
                task=task,
                top_k=top_k,
                budget=budget,
            )

            validation_badge = "PASSED" if proposal.is_valid else "FAILED"
            first_p = proposal.proposals[0] if proposal.proposals else None
            diff_text = first_p.unified_diff if first_p else ""
            reasoning = (
                first_p.architectural_reasoning if first_p else proposal.verification_summary
            )
            return (
                f"### Test Proposal for `{target_file}`\n"
                f"- **AST Validation**: `{validation_badge}`\n\n"
                f"#### Test Strategy & Reasoning\n{reasoning}\n\n"
                f"#### Proposed Unified Diff\n"
                f"```diff\n{diff_text}\n```"
            )
        except Exception as e:
            logger.exception("Error in propose_tests: %s", e)
            return f"Error generating test proposal: {e}"

    # -------------------------------------------------------------------------
    # Tool 8: inspect_dependencies
    # -------------------------------------------------------------------------
    @server.tool()
    def inspect_dependencies(manifest_file: Optional[str] = None) -> str:
        """Inspect repository dependency manifests (pyproject.toml, requirements.txt, package.json).

        Args:
            manifest_file: Optional specific manifest path. If omitted, scans common root manifests.
        """
        try:
            repo = be.require_active_repo()
            assert be.tool_registry is not None
            if manifest_file:
                DiffEngine.validate_target_path(repo, manifest_file)
            args = {"manifest_file": manifest_file} if manifest_file else {}
            res = be.tool_registry._handle_inspect_dependencies(args)
            return f"### Dependency Manifest Inspection\n\n```\n{res}\n```"
        except Exception as e:
            logger.exception("Error in inspect_dependencies: %s", e)
            return f"Error inspecting dependencies: {e}"

    # -------------------------------------------------------------------------
    # Tool 9: investigate_task
    # -------------------------------------------------------------------------
    @server.tool()
    def investigate_task(
        task: str,
        max_iterations: int = 5,
        max_steps: int = 10,
        budget: int = 8192,
    ) -> str:
        """Run an autonomous multi-turn ReAct investigation loop to analyze a problem or locate relevant code.

        Args:
            task: Engineering investigation goal (e.g., 'Trace how user session tokens are verified').
            max_iterations: Maximum tool action iterations (default: 5).
            max_steps: Maximum reasoning steps (default: 10).
            budget: Cumulative context token budget.
        """
        try:
            be.require_active_repo()
            assert be.agent_coordinator is not None
            be.agent_coordinator.config.max_iterations = max_iterations
            be.agent_coordinator.config.max_reasoning_steps = max_steps
            be.agent_coordinator.config.token_budget = budget

            response = be.agent_coordinator.run(task=task)

            steps_summary = []
            for s in response.steps:
                tool_info = f" -> `{s.tool_call.tool_name}`" if s.tool_call else ""
                steps_summary.append(f"- Step {s.step_number}{tool_info}: {s.thought[:140]}...")

            steps_md = "\n".join(steps_summary) if steps_summary else "No intermediate steps."

            proposals_md = ""
            if response.proposals:
                proposals_md = f"\n\n**Generated Proposals ({len(response.proposals)}):**\n"
                for p in response.proposals:
                    proposals_md += f"- Target: `{p.get('target_file')}` ({len(p.get('unified_diff', '').splitlines())} diff lines)\n"

            return (
                f"### Investigation Report\n"
                f"- **Goal**: {task}\n"
                f"- **Status**: `{response.stopping_reason.value}` (Total steps: {response.total_steps})\n\n"
                f"#### Trajectory Summary\n{steps_md}\n\n"
                f"#### Final Findings\n{response.final_answer}{proposals_md}"
            )
        except Exception as e:
            logger.exception("Error in investigate_task: %s", e)
            return f"Error during investigation: {e}"

    # -------------------------------------------------------------------------
    # Tool 10: repair_code
    # -------------------------------------------------------------------------
    @server.tool()
    def repair_code(
        task: str,
        test_command: str,
        max_retries: int = 3,
        max_steps: int = 10,
        budget: int = 8192,
        workflow_timeout: float = 180.0,
    ) -> str:
        """Orchestrate an end-to-end self-repair workflow: plan, propose, sandbox test, diagnose, and retry.

        Tests are executed strictly inside a disposable isolated sandbox with loopback proxy isolation.
        The target repository is never modified directly.

        Args:
            task: Description of the bug to fix or feature to implement.
            test_command: Authorized test runner command (e.g. 'pytest tests/test_api.py').
            max_retries: Maximum repair iterations upon test failure (default: 3).
            max_steps: Cumulative reasoning step limit.
            budget: Cumulative token budget.
            workflow_timeout: Cumulative workflow timeout in seconds (default: 180.0).
        """
        try:
            repo = be.require_active_repo()
            assert be.retriever is not None

            # Execute repair strictly inside an isolated disposable workspace
            with tempfile.TemporaryDirectory(prefix="repair_sandbox_") as sandbox_dir:
                sandbox_path = Path(sandbox_dir).resolve()
                shutil.copytree(
                    repo,
                    sandbox_path,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(
                        ".git", ".venv", "__pycache__", ".pytest_cache", ".codebase_assistant"
                    ),
                )

                sandbox_policy = PolicyEngine(sandbox_path)
                sandbox_perm = PermissionEngine(auto_approve_consequential=True)
                sandbox_code_gen = CodeProposalGenerator(
                    repository_path=sandbox_path,
                    retriever=be.retriever,
                    llm_client=be.llm_client,
                )
                sandbox_test_gen = TestProposalGenerator(
                    repository_path=sandbox_path,
                    retriever=be.retriever,
                    llm_client=be.llm_client,
                )
                sandbox_repair_coord = RepairWorkflowCoordinator(
                    repository_path=sandbox_path,
                    code_generator=sandbox_code_gen,
                    test_generator=sandbox_test_gen,
                    retriever=be.retriever,
                    gemini_client=be.llm_client,
                    policy_engine=sandbox_policy,
                    permission_engine=sandbox_perm,
                    max_retries=max_retries,
                    max_reasoning_steps=max_steps,
                    token_budget=budget,
                    workflow_timeout_seconds=workflow_timeout,
                )

                report = sandbox_repair_coord.run(
                    task=task,
                    test_command=test_command,
                )

                status_icon = (
                    "SUCCESS ✅"
                    if report.status == RepairWorkflowStatus.SUCCESS
                    else f"FAILED ❌ ({report.status.value})"
                )
                diff_block = (
                    f"```diff\n{report.final_diff}\n```"
                    if report.final_diff
                    else "*No patch generated or patch was cleanly rolled back.*"
                )

                return (
                    f"### Repair Workflow Execution Report\n"
                    f"- **Task**: {task}\n"
                    f"- **Test Command**: `{test_command}`\n"
                    f"- **Result**: `{status_icon}`\n"
                    f"- **Attempts**: {report.total_attempts} of {max_retries}\n"
                    f"- **Duration**: {report.total_duration_seconds:.2f}s\n\n"
                    f"#### Architectural Reasoning & Diagnosis\n{report.summary}\n\n"
                    f"#### Final Verified Patch\n{diff_block}\n\n"
                    f"*To apply this patch to a designated writable workspace, use `apply_approved_patch`.*"
                )
        except Exception as e:
            logger.exception("Error in repair_code: %s", e)
            return f"Error executing repair workflow: {e}"

    # -------------------------------------------------------------------------
    # Tool 11: apply_approved_patch
    # -------------------------------------------------------------------------
    @server.tool()
    def apply_approved_patch(
        patch_content: str,
        target_workspace: Optional[str] = None,
        auth_token: Optional[str] = None,
        confirmation: Optional[str] = None,
    ) -> str:
        """Apply a validated unified diff patch to an explicitly designated writable workspace.

        CONSEQUENTIAL TOOL: Requires explicit human authorization. Target repositories are protected
        and read-only by default. Over conversational MCP, model-generated confirmations (such as
        confirmation='CONFIRM_APPLY') cannot authorize direct file modifications on disk.
        When authorization is missing or unverified, the patch is pre-flight validated and saved
        externally for manual application via git apply.

        Args:
            patch_content: Unified diff text (git diff format).
            target_workspace: Path to the writable development workspace directory.
            auth_token: Optional server-issued one-time authorization token for programmatic application.
            confirmation: Legacy confirmation parameter (model-generated values cannot bypass authorization).
        """
        try:
            if target_workspace:
                target_dir = Path(target_workspace).resolve()
            else:
                repo = be.require_active_repo()
                target_dir = repo.resolve()
            real_target = Path(os.path.realpath(target_dir))
            if target_dir != real_target:
                return f"Error: Symlink escape or path mismatch detected: {target_dir} != {real_target}"

            if not target_dir.is_dir():
                return f"Error: Target workspace is not a valid directory: '{target_dir}'"

            # 1. Protected repository check: protected repositories can NEVER be modified
            if be.is_protected_repository(target_dir):
                return (
                    f"### Patch Application Denied: Protected Repository\n"
                    f"The repository at `{target_dir}` is a protected knowledge source and is strictly read-only.\n"
                    f"Modifying files in protected repositories is prohibited by system policy."
                )

            # 2. Authorized workspace check: must be explicitly registered writable workspace
            if not be.is_workspace_authorized(target_dir):
                return (
                    f"### Patch Application Denied: Unauthorized Workspace\n"
                    f"The directory `{target_dir}` is not a registered writable workspace.\n"
                    f"Only explicitly registered development workspaces can be modified."
                )

            # 3. Parse multi-file diff
            try:
                file_diffs = DiffEngine.split_unified_diff(patch_content)
            except Exception as split_err:
                return f"Error parsing unified diff: {split_err}"

            if not file_diffs:
                return "Error: Invalid unified diff: No target file headers found ('+++ b/...')."

            # 4. Pre-flight validation on all affected files
            preflight_state: Dict[Path, Tuple[Optional[str], int, str, str]] = {}
            # target_path -> (original_text, original_mode, patched_text, rel_file)

            for rel_file, f_diff in file_diffs:
                # Reject path traversal
                if ".." in Path(rel_file).parts or Path(rel_file).is_absolute():
                    return f"Error: Path traversal attempt rejected in diff header: '{rel_file}'"

                try:
                    target_path = DiffEngine.validate_target_path(target_dir, rel_file)
                except ValueError as ve:
                    return f"Error: Security boundary violation for '{rel_file}': {ve}"

                real_file_path = Path(os.path.realpath(target_path))
                if target_path.exists() and real_file_path != target_path:
                    return f"Error: Symlink escape detected for target file: '{target_path}'"

                original_text: Optional[str] = None
                original_mode: int = 0o644
                if target_path.exists():
                    original_text = target_path.read_text(encoding="utf-8")
                    original_mode = target_path.stat().st_mode

                try:
                    patched_text = DiffEngine.apply_diff_in_memory(original_text or "", f_diff)
                except Exception as patch_err:
                    return f"Error: In-memory dry run failed for '{rel_file}': {patch_err}"

                if target_path.suffix == ".py":
                    is_valid, syntax_errors = DiffEngine.validate_ast_syntax(
                        patched_text, filename=rel_file
                    )
                    if not is_valid:
                        return f"Error: Pre-flight AST syntax check failed for '{rel_file}': {'; '.join(syntax_errors)}"

                preflight_state[target_path] = (
                    original_text,
                    original_mode,
                    patched_text,
                    rel_file,
                )

            # 5. Check authorization token
            token_authorized = be.verify_and_consume_token(auth_token, target_dir)
            if not token_authorized:
                # Model-generated confirmation ("CONFIRM_APPLY") alone cannot bypass authorization.
                # Save validated patch externally for safe manual application.
                patch_dir = (
                    be.cache_dir / "patches"
                    if be.cache_dir
                    else Path(tempfile.gettempdir()) / "codebase_assistant_patches"
                )
                patch_dir.mkdir(parents=True, exist_ok=True)
                saved_patch = patch_dir / f"validated_patch_{secrets.token_hex(4)}.diff"
                saved_patch.write_text(patch_content, encoding="utf-8")

                file_list = "\n".join(
                    f"- `{rel_f}`" for _, (_, _, _, rel_f) in preflight_state.items()
                )
                return (
                    f"### Pre-Flight Validation Succeeded: Manual Application Required\n\n"
                    f"All in-memory patch dry-runs and Python AST syntax checks PASSED for:\n{file_list}\n\n"
                    f"**Authorization Policy**: Over conversational MCP, model-generated confirmations "
                    f"cannot authorize direct disk modifications.\n"
                    f"The pre-flight validated patch has been saved to:\n`{saved_patch}`\n\n"
                    f"To apply this patch to your workspace, run:\n"
                    f"```bash\ncd {target_dir} && git apply {saved_patch}\n```"
                )

            # 6. Authorized programmatic application with atomic rollback
            written_stack: List[Tuple[Path, Optional[str], int, bool]] = []
            # (target_path, orig_text, orig_mode, is_new)
            try:
                for target_path, (
                    orig_text,
                    orig_mode,
                    patched_text,
                    rel_file,
                ) in preflight_state.items():
                    is_new = orig_text is None and not target_path.exists()
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(patched_text, encoding="utf-8")
                    written_stack.append((target_path, orig_text, orig_mode, is_new))
            except Exception as write_err:
                logger.error(
                    "Write error during patch application: %s. Initiating atomic rollback.",
                    write_err,
                )
                rollback_errors = []
                for target_path, orig_text, orig_mode, is_new in reversed(written_stack):
                    try:
                        if not is_new and orig_text is not None:
                            target_path.write_text(orig_text, encoding="utf-8")
                            try:
                                target_path.chmod(orig_mode)
                            except Exception:
                                pass
                        elif is_new and target_path.exists():
                            target_path.unlink()
                    except Exception as rb_ex:
                        rollback_errors.append(f"{target_path}: {rb_ex}")

                if rollback_errors:
                    return f"CRITICAL: ROLLBACK_FAILED after write error ({write_err}): {'; '.join(rollback_errors)}"
                return f"Error writing patch: {write_err}. Workspace cleanly rolled back to original state."

            patched_names = [rel_f for _, (_, _, _, rel_f) in preflight_state.items()]
            return (
                f"### Patch Applied Successfully ✅\n"
                f"- **Target Workspace**: `{target_dir}`\n"
                f"- **Modified Files ({len(patched_names)})**:\n"
                + "\n".join(f"  - `{f}`" for f in patched_names)
                + f"\n\nAll modified files passed in-memory dry-runs and Python AST syntax checks."
            )
        except Exception as e:
            logger.exception("Error in apply_approved_patch: %s", e)
            return f"Error applying patch: {e}"

    return server


def main() -> None:
    """Run the Codebase Assistant MCP server over STDIO transport."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logger.info("Starting Codebase Assistant MCP Server...")
    server = create_mcp_server()
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
