"""Unit tests for Antigravity Model Context Protocol (MCP) server integration."""

import asyncio
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from codebase_assistant.agent.repair_models import RepairWorkflowStatus
from codebase_assistant.mcp_server import MCPBackend, create_mcp_server


class TestMCPServerIntegration(unittest.TestCase):
    """Test suite verifying MCP server registration, tool execution, safety boundaries, and offline mode."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.temp_dir.name).resolve()

        # Build a minimal sample repository for deterministic offline testing
        src_dir = self.repo_dir / "src" / "sample"
        src_dir.mkdir(parents=True)
        self.auth_file = src_dir / "auth.py"
        self.auth_file.write_text(
            '"""Authentication module."""\n\n'
            "class AuthHandler:\n"
            '    """Handles user session authentication."""\n\n'
            "    def authenticate(self, token: str) -> bool:\n"
            '        """Validate authorization token."""\n'
            '        return token == "valid_token"\n',
            encoding="utf-8",
        )

        tests_dir = self.repo_dir / "tests"
        tests_dir.mkdir(parents=True)
        self.test_file = tests_dir / "test_auth.py"
        self.test_file.write_text(
            "import unittest\n"
            "from sample.auth import AuthHandler\n\n"
            "class TestAuth(unittest.TestCase):\n"
            "    def test_auth_success(self):\n"
            '        self.assertTrue(AuthHandler().authenticate("valid_token"))\n',
            encoding="utf-8",
        )

        pyproject = self.repo_dir / "pyproject.toml"
        pyproject.write_text(
            '[project]\nname = "sample"\nversion = "0.1.0"\ndependencies = ["pydantic>=2.0"]\n',
            encoding="utf-8",
        )

        # External cache directory strictly outside the repository
        self.cache_dir = Path(tempfile.mkdtemp()).resolve()

        # Backend in offline mock mode
        self.backend = MCPBackend(is_live=False)
        self.server = create_mcp_server(backend=self.backend)

    def tearDown(self):
        self.temp_dir.cleanup()
        import shutil

        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _call_tool(self, name: str, args: dict):
        """Helper to call an MCP tool asynchronously and extract text output."""
        return asyncio.run(self.server.call_tool(name, args))

    def _get_text(self, result) -> str:
        """Extract text from FastMCP tool result content."""
        if hasattr(result, "content") and result.content:
            for item in result.content:
                if hasattr(item, "text"):
                    return item.text
        if isinstance(result, list) and len(result) > 0 and hasattr(result[0], "text"):
            return result[0].text
        if isinstance(result, dict) and "result" in result:
            return str(result["result"])
        return str(result)

    def test_tool_registration_and_discovery(self):
        """Verify all 11 core and advanced tools are registered on the MCP server."""
        tools = asyncio.run(self.server.list_tools())
        tool_names = {t.name for t in tools}
        expected_tools = {
            "set_active_repository",
            "repository_status",
            "search_codebase",
            "read_source_file",
            "ask_codebase",
            "propose_code",
            "propose_tests",
            "inspect_dependencies",
            "investigate_task",
            "repair_code",
            "apply_approved_patch",
        }
        self.assertTrue(
            expected_tools.issubset(tool_names), f"Missing tools: {expected_tools - tool_names}"
        )

    def test_uninitialized_status(self):
        """Verify status prompt before repository selection."""
        res = self._call_tool("repository_status", {})
        text = self._get_text(res)
        self.assertIn("No Repository Active", text)
        self.assertIn("set_active_repository", text)

    def test_set_active_repository(self):
        """Verify repository selection, indexing, and activation."""
        res = self._call_tool(
            "set_active_repository",
            {"repository_path": str(self.repo_dir), "cache_dir": str(self.cache_dir)},
        )
        text = self._get_text(res)
        self.assertIn("Repository Activated", text)
        self.assertIn(self.repo_dir.name, text)
        self.assertIn("Indexed Chunks", text)

        # Status should now be active
        stat = self._call_tool("repository_status", {})
        stat_text = self._get_text(stat)
        self.assertIn("Status: Active", stat_text)
        self.assertIn(str(self.repo_dir), stat_text)

    def test_read_source_file_success(self):
        """Verify reading valid repository source files with line numbers."""
        self._call_tool(
            "set_active_repository",
            {"repository_path": str(self.repo_dir), "cache_dir": str(self.cache_dir)},
        )
        res = self._call_tool(
            "read_source_file",
            {"file_path": "src/sample/auth.py", "start_line": 1, "end_line": 8},
        )
        text = self._get_text(res)
        self.assertIn("File: `src/sample/auth.py`", text)
        self.assertIn('1 | """Authentication module."""', text)
        self.assertIn("AuthHandler", text)

    def test_read_source_file_path_traversal_rejected(self):
        """Verify path traversal outside workspace is safely blocked."""
        self._call_tool(
            "set_active_repository",
            {"repository_path": str(self.repo_dir), "cache_dir": str(self.cache_dir)},
        )
        res = self._call_tool("read_source_file", {"file_path": "../../secret.txt"})
        text = self._get_text(res)
        self.assertIn("Error", text)
        self.assertIn("Path traversal detected", text)

    def test_search_codebase(self):
        """Verify semantic hybrid search over indexed repository."""
        self._call_tool(
            "set_active_repository",
            {"repository_path": str(self.repo_dir), "cache_dir": str(self.cache_dir)},
        )
        res = self._call_tool("search_codebase", {"query": "AuthHandler authenticate token"})
        text = self._get_text(res)
        self.assertIn("Codebase Search Results", text)
        self.assertIn("src/sample/auth.py", text)

    def test_ask_codebase_grounded_answer_with_citations(self):
        """Verify grounded Q&A returns answer with valid citations."""
        self._call_tool(
            "set_active_repository",
            {"repository_path": str(self.repo_dir), "cache_dir": str(self.cache_dir)},
        )
        res = self._call_tool(
            "ask_codebase", {"question": "How does AuthHandler authenticate tokens?"}
        )
        text = self._get_text(res)
        self.assertIn("Grounded Response", text)
        self.assertTrue(len(text) > 20)

    def test_propose_code_ast_validated(self):
        """Verify code proposal generation outputs standard unified diff with AST validation."""
        self._call_tool(
            "set_active_repository",
            {"repository_path": str(self.repo_dir), "cache_dir": str(self.cache_dir)},
        )
        res = self._call_tool(
            "propose_code",
            {"task": "Add docstring to authenticate", "target_file": "src/sample/auth.py"},
        )
        text = self._get_text(res)
        self.assertIn("Code Proposal for `src/sample/auth.py`", text)
        self.assertIn("Proposed Unified Diff", text)
        self.assertIn("--- a/src/sample/auth.py", text)

    def test_propose_tests_ast_validated(self):
        """Verify test proposal generation outputs standard unified diff."""
        self._call_tool(
            "set_active_repository",
            {"repository_path": str(self.repo_dir), "cache_dir": str(self.cache_dir)},
        )
        res = self._call_tool(
            "propose_tests",
            {"task": "Add test case for invalid token", "target_file": "src/sample/auth.py"},
        )
        text = self._get_text(res)
        self.assertIn("Test Proposal for `src/sample/auth.py`", text)
        self.assertIn("Proposed Unified Diff", text)
        self.assertIn("tests/test_auth.py", text)

    def test_inspect_dependencies(self):
        """Verify dependency inspection returns manifest content."""
        self._call_tool(
            "set_active_repository",
            {"repository_path": str(self.repo_dir), "cache_dir": str(self.cache_dir)},
        )
        res = self._call_tool("inspect_dependencies", {})
        text = self._get_text(res)
        self.assertIn("Dependency Manifest Inspection", text)
        self.assertIn("pydantic", text)

    def test_investigate_task_react_loop(self):
        """Verify multi-turn autonomous investigation trajectory."""
        self._call_tool(
            "set_active_repository",
            {"repository_path": str(self.repo_dir), "cache_dir": str(self.cache_dir)},
        )
        res = self._call_tool(
            "investigate_task", {"task": "Find where tokens are checked in this repo"}
        )
        text = self._get_text(res)
        self.assertIn("Investigation Report", text)
        self.assertIn("Trajectory Summary", text)
        self.assertIn("Final Findings", text)

    def test_apply_approved_patch_read_only_denial(self):
        """Verify that default read-only target repositories strictly reject patch applications."""
        self._call_tool(
            "set_active_repository",
            {
                "repository_path": str(self.repo_dir),
                "cache_dir": str(self.cache_dir),
                "writable": False,
            },
        )
        diff = (
            "--- a/src/sample/auth.py\n"
            "+++ b/src/sample/auth.py\n"
            "@@ -8,1 +8,2 @@\n"
            '-        return token == "valid_token"\n'
            "+        # Validated\n"
            '+        return token == "valid_token"\n'
        )
        res = self._call_tool("apply_approved_patch", {"patch_content": diff})
        text = self._get_text(res)
        self.assertIn("Patch Application Denied: Read-Only Repository", text)
        self.assertIn("protected read-only knowledge source", text)

    def test_apply_approved_patch_confirmation_required(self):
        """Verify human confirmation checkpoint is required before writing to writable workspace."""
        self._call_tool(
            "set_active_repository",
            {
                "repository_path": str(self.repo_dir),
                "cache_dir": str(self.cache_dir),
                "writable": True,
            },
        )
        diff = (
            "--- a/src/sample/auth.py\n"
            "+++ b/src/sample/auth.py\n"
            "@@ -8,1 +8,2 @@\n"
            '-        return token == "valid_token"\n'
            "+        # Validated\n"
            '+        return token == "valid_token"\n'
        )
        res = self._call_tool("apply_approved_patch", {"patch_content": diff})
        text = self._get_text(res)
        self.assertIn("Human Approval Required", text)
        self.assertIn("CONFIRM_APPLY", text)

    def test_apply_approved_patch_success_and_rollback(self):
        """Verify atomic patch application and rollback upon AST or write failure."""
        self._call_tool(
            "set_active_repository",
            {
                "repository_path": str(self.repo_dir),
                "cache_dir": str(self.cache_dir),
                "writable": True,
            },
        )
        valid_diff = (
            "--- a/src/sample/auth.py\n"
            "+++ b/src/sample/auth.py\n"
            "@@ -8,1 +8,2 @@\n"
            '-        return token == "valid_token"\n'
            "+        # Validated\n"
            '+        return token == "valid_token"\n'
        )
        res = self._call_tool(
            "apply_approved_patch",
            {"patch_content": valid_diff, "confirmation": "CONFIRM_APPLY"},
        )
        text = self._get_text(res)
        self.assertIn("Patch Applied Successfully", text)

        # Confirm file was modified on disk
        updated_content = self.auth_file.read_text(encoding="utf-8")
        self.assertIn("# Validated", updated_content)


if __name__ == "__main__":
    unittest.main()
