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

    def test_apply_approved_patch_unauthorized_workspace_denial(self):
        """Verify that default read-only or unregistered target repositories strictly reject patch applications."""
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
        self.assertIn("Patch Application Denied: Unauthorized Workspace", text)

    def test_apply_approved_patch_model_confirmation_cannot_bypass_authorization(self):
        """Negative test: model-generated confirmation ('CONFIRM_APPLY') alone cannot modify files on disk."""
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
            "+        # Model modified\n"
            '+        return token == "valid_token"\n'
        )
        res = self._call_tool(
            "apply_approved_patch",
            {"patch_content": diff, "confirmation": "CONFIRM_APPLY"},
        )
        text = self._get_text(res)
        self.assertIn("Pre-Flight Validation Succeeded: Manual Application Required", text)
        self.assertIn(
            "model-generated confirmations cannot authorize direct disk modifications", text
        )
        self.assertIn("git apply", text)

        # Crucial invariant: disk file must NOT have changed!
        content = self.auth_file.read_text(encoding="utf-8")
        self.assertNotIn("# Model modified", content)

    def test_apply_approved_patch_with_valid_token(self):
        """Verify programmatic write succeeds when a valid server authorization token is provided."""
        self._call_tool(
            "set_active_repository",
            {
                "repository_path": str(self.repo_dir),
                "cache_dir": str(self.cache_dir),
                "writable": True,
            },
        )
        token = self.backend.issue_authorization_token(self.repo_dir)
        valid_diff = (
            "--- a/src/sample/auth.py\n"
            "+++ b/src/sample/auth.py\n"
            "@@ -8,1 +8,2 @@\n"
            '-        return token == "valid_token"\n'
            "+        # Token authorized write\n"
            '+        return token == "valid_token"\n'
        )
        res = self._call_tool(
            "apply_approved_patch",
            {"patch_content": valid_diff, "auth_token": token},
        )
        text = self._get_text(res)
        self.assertIn("Patch Applied Successfully", text)

        # Confirm file was modified on disk
        updated_content = self.auth_file.read_text(encoding="utf-8")
        self.assertIn("# Token authorized write", updated_content)

    def test_workspace_authorization_protected_repo_denial(self):
        """Verify protected repositories (e.g. ai-evaluation) can never be registered as writable or patched."""
        with tempfile.TemporaryDirectory() as p_dir:
            protected_repo = Path(p_dir) / "ai_evaluation"
            protected_repo.mkdir()
            # Direct registration attempt must raise PermissionError
            with self.assertRaises(PermissionError):
                self.backend.register_writable_workspace(protected_repo)

            # Check via tool
            res = self._call_tool(
                "apply_approved_patch",
                {
                    "patch_content": "--- a/x.py\n+++ b/x.py\n@@ -0,0 +1,1 @@\n+print(1)\n",
                    "target_workspace": str(protected_repo),
                },
            )
            text = self._get_text(res)
            self.assertIn("Protected Repository", text)

    def test_workspace_authorization_unregistered_workspace_denial(self):
        """Verify unregistered external workspace is rejected."""
        self._call_tool(
            "set_active_repository",
            {
                "repository_path": str(self.repo_dir),
                "cache_dir": str(self.cache_dir),
                "writable": True,
            },
        )
        with tempfile.TemporaryDirectory() as other_dir:
            diff = "--- a/foo.py\n+++ b/foo.py\n@@ -0,0 +1,1 @@\n+# foo\n"
            res = self._call_tool(
                "apply_approved_patch",
                {"patch_content": diff, "target_workspace": other_dir},
            )
            text = self._get_text(res)
            self.assertIn("Unauthorized Workspace", text)

    def test_workspace_authorization_path_traversal_denial(self):
        """Verify parent directory traversal in patch diff headers is rejected."""
        self._call_tool(
            "set_active_repository",
            {
                "repository_path": str(self.repo_dir),
                "cache_dir": str(self.cache_dir),
                "writable": True,
            },
        )
        diff = "--- a/../../secret.py\n+++ b/../../secret.py\n@@ -0,0 +1,1 @@\n+# pwned\n"
        res = self._call_tool("apply_approved_patch", {"patch_content": diff})
        text = self._get_text(res)
        self.assertIn("Path traversal attempt rejected", text)

    def test_workspace_authorization_symlink_escape_denial(self):
        """Verify symlinks pointing outside authorized workspace are rejected."""
        self._call_tool(
            "set_active_repository",
            {
                "repository_path": str(self.repo_dir),
                "cache_dir": str(self.cache_dir),
                "writable": True,
            },
        )
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_file = Path(outside_dir) / "external.py"
            outside_file.write_text("# external\n", encoding="utf-8")
            symlink_file = self.repo_dir / "src" / "sample" / "linked.py"
            try:
                symlink_file.symlink_to(outside_file)
            except OSError:
                return

            diff = (
                "--- a/src/sample/linked.py\n"
                "+++ b/src/sample/linked.py\n"
                "@@ -1,1 +1,2 @@\n"
                "-# external\n"
                "+# compromised\n"
            )
            res = self._call_tool("apply_approved_patch", {"patch_content": diff})
            text = self._get_text(res)
            self.assertTrue(
                "symlink escape" in text.lower() or "security boundary violation" in text.lower(),
                f"Expected symlink escape rejection, got: {text}",
            )

    def test_workspace_unregistered_after_registration(self):
        """Verify workspace path rejected if unregistered after registration."""
        ws = self.backend.register_writable_workspace(self.repo_dir)
        self.assertTrue(self.backend.is_workspace_authorized(ws))
        self.backend.unregister_writable_workspace(ws)
        self.assertFalse(self.backend.is_workspace_authorized(ws))

    def test_multi_file_patch_success(self):
        """Verify multi-file patch applies across multiple files with valid token."""
        self._call_tool(
            "set_active_repository",
            {
                "repository_path": str(self.repo_dir),
                "cache_dir": str(self.cache_dir),
                "writable": True,
            },
        )
        token = self.backend.issue_authorization_token(self.repo_dir)
        multi_diff = (
            "--- a/src/sample/auth.py\n"
            "+++ b/src/sample/auth.py\n"
            "@@ -8,1 +8,2 @@\n"
            '-        return token == "valid_token"\n'
            "+        # File 1 updated\n"
            '+        return token == "valid_token"\n'
            "--- a/tests/test_auth.py\n"
            "+++ b/tests/test_auth.py\n"
            "@@ -5,2 +5,3 @@\n"
            "     def test_auth_success(self):\n"
            "+        # File 2 updated\n"
            '         self.assertTrue(AuthHandler().authenticate("valid_token"))\n'
        )
        res = self._call_tool(
            "apply_approved_patch", {"patch_content": multi_diff, "auth_token": token}
        )
        text = self._get_text(res)
        self.assertIn("Patch Applied Successfully", text)
        self.assertIn("src/sample/auth.py", text)
        self.assertIn("tests/test_auth.py", text)
        self.assertIn("# File 1 updated", self.auth_file.read_text(encoding="utf-8"))
        self.assertIn("# File 2 updated", self.test_file.read_text(encoding="utf-8"))

    def test_multi_file_patch_invalid_second_file_zero_partial_write(self):
        """Verify that when second file fails pre-flight AST validation, first file is NOT modified."""
        self._call_tool(
            "set_active_repository",
            {
                "repository_path": str(self.repo_dir),
                "cache_dir": str(self.cache_dir),
                "writable": True,
            },
        )
        token = self.backend.issue_authorization_token(self.repo_dir)
        multi_diff = (
            "--- a/src/sample/auth.py\n"
            "+++ b/src/sample/auth.py\n"
            "@@ -8,1 +8,2 @@\n"
            '-        return token == "valid_token"\n'
            "+        # File 1 should not change\n"
            '+        return token == "valid_token"\n'
            "--- a/tests/test_auth.py\n"
            "+++ b/tests/test_auth.py\n"
            "@@ -5,2 +5,3 @@\n"
            "     def test_auth_success(self):\n"
            "+        def def syntax_error(\n"
            '         self.assertTrue(AuthHandler().authenticate("valid_token"))\n'
        )
        res = self._call_tool(
            "apply_approved_patch", {"patch_content": multi_diff, "auth_token": token}
        )
        text = self._get_text(res)
        self.assertIn("Pre-flight AST syntax check failed", text)
        # Crucial: file 1 was NOT modified!
        self.assertNotIn("# File 1 should not change", self.auth_file.read_text(encoding="utf-8"))

    def test_atomic_rollback_on_write_failure(self):
        """Verify atomic rollback restores original files and file permissions when write fails."""
        self._call_tool(
            "set_active_repository",
            {
                "repository_path": str(self.repo_dir),
                "cache_dir": str(self.cache_dir),
                "writable": True,
            },
        )
        token = self.backend.issue_authorization_token(self.repo_dir)
        multi_diff = (
            "--- a/src/sample/auth.py\n"
            "+++ b/src/sample/auth.py\n"
            "@@ -8,1 +8,2 @@\n"
            '-        return token == "valid_token"\n'
            "+        # File 1 rollback test\n"
            '+        return token == "valid_token"\n'
            "--- a/tests/test_auth.py\n"
            "+++ b/tests/test_auth.py\n"
            "@@ -5,2 +5,3 @@\n"
            "     def test_auth_success(self):\n"
            "+        # File 2\n"
            '         self.assertTrue(AuthHandler().authenticate("valid_token"))\n'
        )
        orig_auth = self.auth_file.read_text(encoding="utf-8")
        orig_mode = self.auth_file.stat().st_mode

        # Make second file read-only to trigger write failure
        self.test_file.chmod(0o444)
        try:
            res = self._call_tool(
                "apply_approved_patch", {"patch_content": multi_diff, "auth_token": token}
            )
            text = self._get_text(res)
            self.assertIn("Workspace cleanly rolled back", text)
        finally:
            self.test_file.chmod(0o644)

        # Invariant: auth_file must match exact original content and mode
        self.assertEqual(self.auth_file.read_text(encoding="utf-8"), orig_auth)
        self.assertEqual(self.auth_file.stat().st_mode, orig_mode)

    def test_atomic_rollback_unlinks_newly_created_files(self):
        """Verify that newly created files are unlinked when a multi-file write fails."""
        self._call_tool(
            "set_active_repository",
            {
                "repository_path": str(self.repo_dir),
                "cache_dir": str(self.cache_dir),
                "writable": True,
            },
        )
        token = self.backend.issue_authorization_token(self.repo_dir)
        new_file_diff = (
            "--- /dev/null\n"
            "+++ b/src/sample/new_mod.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+# brand new file\n"
            "+NEW_CONST = 42\n"
            "--- a/tests/test_auth.py\n"
            "+++ b/tests/test_auth.py\n"
            "@@ -5,2 +5,3 @@\n"
            "     def test_auth_success(self):\n"
            "+        # File 2\n"
            '         self.assertTrue(AuthHandler().authenticate("valid_token"))\n'
        )
        new_path = self.repo_dir / "src" / "sample" / "new_mod.py"

        # Make second file read-only to trigger write failure after creating new file
        self.test_file.chmod(0o444)
        try:
            res = self._call_tool(
                "apply_approved_patch", {"patch_content": new_file_diff, "auth_token": token}
            )
            text = self._get_text(res)
            self.assertIn("Workspace cleanly rolled back", text)
        finally:
            self.test_file.chmod(0o644)

        # Invariant: new_path must not exist on disk
        self.assertFalse(new_path.exists())

    def test_critical_rollback_failure_reporting(self):
        """Verify CRITICAL: ROLLBACK_FAILED is returned when rollback encounters an error."""
        self._call_tool(
            "set_active_repository",
            {
                "repository_path": str(self.repo_dir),
                "cache_dir": str(self.cache_dir),
                "writable": True,
            },
        )
        token = self.backend.issue_authorization_token(self.repo_dir)
        multi_diff = (
            "--- a/src/sample/auth.py\n"
            "+++ b/src/sample/auth.py\n"
            "@@ -8,1 +8,2 @@\n"
            '-        return token == "valid_token"\n'
            "+        # File 1\n"
            '+        return token == "valid_token"\n'
            "--- a/tests/test_auth.py\n"
            "+++ b/tests/test_auth.py\n"
            "@@ -5,2 +5,3 @@\n"
            "     def test_auth_success(self):\n"
            "+        # File 2\n"
            '         self.assertTrue(AuthHandler().authenticate("valid_token"))\n'
        )
        with patch.object(
            Path,
            "write_text",
            side_effect=[
                None,
                OSError("Write failed on file 2"),
                OSError("Disk locked during rollback"),
            ],
        ):
            res = self._call_tool(
                "apply_approved_patch", {"patch_content": multi_diff, "auth_token": token}
            )
            text = self._get_text(res)
            self.assertIn("CRITICAL: ROLLBACK_FAILED", text)

    def test_repair_code_sandbox_isolation(self):
        """Verify repair_code executes in disposable temporary directory without mutating target repository."""
        self._call_tool(
            "set_active_repository",
            {
                "repository_path": str(self.repo_dir),
                "cache_dir": str(self.cache_dir),
                "writable": False,
            },
        )
        orig_auth = self.auth_file.read_text(encoding="utf-8")
        orig_test = self.test_file.read_text(encoding="utf-8")

        res = self._call_tool(
            "repair_code",
            {
                "task": "Add invalid token handling",
                "test_command": "python -m unittest tests/test_auth.py",
            },
        )
        text = self._get_text(res)
        self.assertIn("Repair Workflow Execution Report", text)

        # Invariant: self.repo_dir was NOT modified at all!
        self.assertEqual(self.auth_file.read_text(encoding="utf-8"), orig_auth)
        self.assertEqual(self.test_file.read_text(encoding="utf-8"), orig_test)

    def test_fail_closed_provider_initialization(self):
        """Verify live mode without --allow-fallback fails closed when credentials fail."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=True):
            with patch(
                "codebase_assistant.mcp_server.GeminiClient",
                side_effect=ValueError("No API key"),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    MCPBackend(is_live=True, allow_fallback=False)
                self.assertIn("Failed to initialize live Gemini providers", str(ctx.exception))

            # With allow_fallback=True, it falls back to mock mode
            with patch(
                "codebase_assistant.mcp_server.GeminiClient",
                side_effect=ValueError("No API key"),
            ):
                be = MCPBackend(is_live=True, allow_fallback=True)
                self.assertFalse(be.is_live)


if __name__ == "__main__":
    unittest.main()
