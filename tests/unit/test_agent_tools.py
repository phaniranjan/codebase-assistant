"""Unit tests for ToolRegistry and tool implementations."""

from pathlib import Path
import shutil
import tempfile
import unittest

from codebase_assistant.agent.models import ToolCall, ToolType
from codebase_assistant.agent.permissions import PermissionEngine
from codebase_assistant.agent.policy import PolicyEngine
from codebase_assistant.agent.tools import ToolRegistry
from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.generation.code_generator import CodeProposalGenerator
from codebase_assistant.generation.test_generator import TestProposalGenerator
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.llm.mock_client import MockGeminiClient
from codebase_assistant.retrieval.hybrid import HybridRetriever


class TestToolRegistry(unittest.TestCase):
    """Test suite verifying all 8 registered tools and their handlers."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.tmp_dir.name)

        # Setup sample files
        src_dir = self.repo_dir / "src"
        src_dir.mkdir(parents=True)
        self.calc_file = src_dir / "calc.py"
        self.calc_file.write_text(
            "def add(a: int, b: int) -> int:\n    return a + b\n\ndef sub(a: int, b: int) -> int:\n    return a - b\n",
            encoding="utf-8",
        )

        tests_dir = self.repo_dir / "tests" / "unit"
        tests_dir.mkdir(parents=True)
        self.test_calc_file = tests_dir / "test_calc.py"
        self.test_calc_file.write_text(
            "import unittest\n\nclass TestCalc(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(1 + 1, 2)\n",
            encoding="utf-8",
        )

        (self.repo_dir / "requirements.txt").write_text("pytest>=7.0.0\nclick>=8.0.0\n", encoding="utf-8")

        # Indexing and Retriever
        self.cache_dir = Path(tempfile.mkdtemp())
        indexer = IncrementalIndexer(self.repo_dir, cache_dir=self.cache_dir)
        indexer.index()
        chunks = indexer.load_chunks()

        provider = MockEmbeddingProvider(dimension=64)
        manager = EmbeddingManager(self.repo_dir, cache_dir=self.cache_dir, provider=provider)
        manager.sync()
        vector_store = manager.load_vector_store()

        self.retriever = HybridRetriever(
            chunks=chunks,
            vector_store=vector_store,
            embedding_provider=provider,
        )
        self.mock_client = MockGeminiClient()

        self.code_gen = CodeProposalGenerator(self.repo_dir, self.retriever, self.mock_client)
        self.test_gen = TestProposalGenerator(self.repo_dir, self.retriever, self.mock_client)
        self.policy = PolicyEngine(self.repo_dir)
        self.perm_engine = PermissionEngine(auto_approve_consequential=True)

        self.registry = ToolRegistry(
            repository_path=self.repo_dir,
            retriever=self.retriever,
            code_generator=self.code_gen,
            test_generator=self.test_gen,
            policy_engine=self.policy,
            permission_engine=self.perm_engine,
            command_timeout_seconds=10.0,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_tool_definitions_count(self):
        defs = self.registry.get_tool_definitions()
        self.assertEqual(len(defs), 8)
        names = {d.name for d in defs}
        expected = {
            "search_codebase",
            "read_source_file",
            "find_related_tests",
            "inspect_dependencies",
            "generate_code_proposal",
            "generate_test_proposal",
            "run_selected_tests",
            "apply_patch",
        }
        self.assertEqual(names, expected)

    def test_search_codebase(self):
        call = ToolCall(tool_name="search_codebase", arguments={"query": "add"})
        res = self.registry.execute_tool(call)
        self.assertTrue(res.success)
        self.assertIn("calc.py", res.output)
        self.assertIn("def add", res.output)

    def test_read_source_file_full_and_range(self):
        # Full file
        call1 = ToolCall(tool_name="read_source_file", arguments={"file_path": "src/calc.py"})
        res1 = self.registry.execute_tool(call1)
        self.assertTrue(res1.success)
        self.assertIn("1: def add", res1.output)

        # Line range
        call2 = ToolCall(tool_name="read_source_file", arguments={"file_path": "src/calc.py", "start_line": 3, "end_line": 5})
        res2 = self.registry.execute_tool(call2)
        self.assertTrue(res2.success)
        self.assertIn("def sub", res2.output)
        self.assertNotIn("def add", res2.output)

        # Path traversal rejected
        call_bad = ToolCall(tool_name="read_source_file", arguments={"file_path": "../outside.py"})
        res_bad = self.registry.execute_tool(call_bad)
        self.assertFalse(res_bad.success)
        self.assertIn("Security Policy Rejection", res_bad.error)

    def test_find_related_tests(self):
        call = ToolCall(tool_name="find_related_tests", arguments={"symbol_or_path": "calc"})
        res = self.registry.execute_tool(call)
        self.assertTrue(res.success)
        self.assertIn("test_calc.py", res.output)

    def test_inspect_dependencies(self):
        call = ToolCall(tool_name="inspect_dependencies", arguments={})
        res = self.registry.execute_tool(call)
        self.assertTrue(res.success)
        self.assertIn("requirements.txt", res.output)
        self.assertIn("pytest>=7.0.0", res.output)

    def test_generate_code_proposal(self):
        call = ToolCall(tool_name="generate_code_proposal", arguments={"task": "Add multiply function", "target_file": "src/calc.py"})
        res = self.registry.execute_tool(call)
        self.assertTrue(res.success)
        self.assertIn("Code proposal generated", res.output)
        self.assertIn("Valid Python", res.output)
        self.assertIn("+++ b/src/calc.py", res.output)
        self.assertEqual(len(self.registry.collected_proposals), 1)

    def test_generate_test_proposal(self):
        call = ToolCall(tool_name="generate_test_proposal", arguments={"source_file": "src/calc.py"})
        res = self.registry.execute_tool(call)
        self.assertTrue(res.success)
        self.assertIn("Test proposal generated", res.output)
        self.assertIn("Valid Python", res.output)

    def test_run_selected_tests_authorized(self):
        call = ToolCall(
            tool_name="run_selected_tests",
            arguments={"command": "python -m unittest tests/unit/test_calc.py"},
        )
        res = self.registry.execute_tool(call)
        self.assertTrue(res.success)
        self.assertIn("PASSED", res.output)
        self.assertIn("Ran 1 test", res.output)

    def test_run_selected_tests_denied(self):
        # Engine that denies authorization
        denying_engine = PermissionEngine(approval_callback=lambda req: False)
        self.registry.permission_engine = denying_engine

        call = ToolCall(
            tool_name="run_selected_tests",
            arguments={"command": "python -m unittest tests/unit/test_calc.py"},
        )
        res = self.registry.execute_tool(call)
        self.assertFalse(res.success)
        self.assertFalse(res.is_authorized)
        self.assertIn("User denied authorization", res.error)

    def test_apply_patch_authorized(self):
        diff = (
            "--- a/src/calc.py\n"
            "+++ b/src/calc.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def add(a: int, b: int) -> int:\n"
            "+    # Added documentation\n"
            "     return a + b\n"
        )
        call = ToolCall(tool_name="apply_patch", arguments={"unified_diff": diff})
        res = self.registry.execute_tool(call)
        self.assertTrue(res.success)
        self.assertIn("Successfully applied patch", res.output)

        # Confirm file was actually updated on disk
        updated_content = self.calc_file.read_text(encoding="utf-8")
        self.assertIn("# Added documentation", updated_content)

    def test_apply_patch_denied(self):
        denying_engine = PermissionEngine(approval_callback=lambda req: False)
        self.registry.permission_engine = denying_engine

        diff = (
            "--- a/src/calc.py\n"
            "+++ b/src/calc.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-x\n"
            "+y\n"
        )
        call = ToolCall(tool_name="apply_patch", arguments={"unified_diff": diff})
        res = self.registry.execute_tool(call)
        self.assertFalse(res.success)
        self.assertFalse(res.is_authorized)
        self.assertIn("User denied authorization", res.error)

    def test_inspect_dependencies_manifest_symlink_rejected(self):
        # Remove existing requirements.txt
        req_file = self.repo_dir / "requirements.txt"
        if req_file.exists():
            req_file.unlink()

        with tempfile.TemporaryDirectory() as outside_dir:
            secret_file = Path(outside_dir) / "secret_requirements.txt"
            secret_file.write_text("SUPER_SECRET_TOKEN=xyz123\n", encoding="utf-8")

            try:
                req_file.symlink_to(secret_file)
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks not supported in this environment.")

            call = ToolCall(tool_name="inspect_dependencies", arguments={})
            res = self.registry.execute_tool(call)
            self.assertTrue(res.success)
            # Manifest symlink pointing outside must be skipped and never leaked
            self.assertNotIn("SUPER_SECRET_TOKEN", res.output)

    def test_run_selected_tests_disposable_sandbox_isolation(self):
        side_effect_test = self.repo_dir / "tests" / "unit" / "test_side_effect.py"
        side_effect_test.write_text(
            "import unittest\nfrom pathlib import Path\n\n"
            "class TestSideEffect(unittest.TestCase):\n"
            "    def test_create_file(self):\n"
            "        Path('sandbox_artifact.txt').write_text('mutated', encoding='utf-8')\n"
            "        self.assertTrue(True)\n",
            encoding="utf-8",
        )

        call = ToolCall(
            tool_name="run_selected_tests",
            arguments={"command": "python -m unittest tests/unit/test_side_effect.py"},
        )
        res = self.registry.execute_tool(call)
        self.assertTrue(res.success)
        self.assertIn("PASSED", res.output)

        # Confirm target repository was NOT mutated by tests executed in the disposable sandbox
        artifact = self.repo_dir / "sandbox_artifact.txt"
        self.assertFalse(artifact.exists(), "Disposable sandbox failed: side effect leaked into repo directory!")

    def test_apply_patch_multi_file_atomic_success(self):
        multi_diff = (
            "--- a/src/calc.py\n"
            "+++ b/src/calc.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def add(a: int, b: int) -> int:\n"
            "+    # First file modified\n"
            "     return a + b\n"
            "--- /dev/null\n"
            "+++ b/src/helper.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def get_version() -> str:\n"
            "+    return '1.0.0'\n"
        )
        call = ToolCall(tool_name="apply_patch", arguments={"unified_diff": multi_diff})
        res = self.registry.execute_tool(call)
        self.assertTrue(res.success)
        self.assertIn("Successfully applied patch to 2 file(s)", res.output)

        self.assertIn("# First file modified", self.calc_file.read_text(encoding="utf-8"))
        helper_file = self.repo_dir / "src" / "helper.py"
        self.assertTrue(helper_file.exists())
        self.assertIn("def get_version", helper_file.read_text(encoding="utf-8"))

    def test_apply_patch_multi_file_atomic_failure_leaves_file1_unchanged(self):
        calc_original = self.calc_file.read_text(encoding="utf-8")

        # Multi-file diff where File 1 is valid, but File 2 contains syntax error
        bad_multi_diff = (
            "--- a/src/calc.py\n"
            "+++ b/src/calc.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def add(a: int, b: int) -> int:\n"
            "+    # Should not be applied\n"
            "     return a + b\n"
            "--- /dev/null\n"
            "+++ b/src/bad_syntax.py\n"
            "@@ -0,0 +1,2 @@\n"
            "+def broken_syntax(:\n"
            "+    return 42\n"
        )
        call = ToolCall(tool_name="apply_patch", arguments={"unified_diff": bad_multi_diff})
        res = self.registry.execute_tool(call)

        self.assertFalse(res.success)
        self.assertIn("failed AST validation", res.error)

        # Critical atomicity guarantee: File 1 was NEVER touched on disk!
        calc_after = self.calc_file.read_text(encoding="utf-8")
        self.assertEqual(calc_original, calc_after)
        self.assertNotIn("# Should not be applied", calc_after)

        # Bad file was never created on disk
        bad_file = self.repo_dir / "src" / "bad_syntax.py"
        self.assertFalse(bad_file.exists())


if __name__ == "__main__":
    unittest.main()
