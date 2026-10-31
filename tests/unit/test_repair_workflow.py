"""Unit tests for RepairWorkflowCoordinator end-to-end task execution and bounded self-repair."""

from pathlib import Path
import shutil
import tempfile
import unittest

from codebase_assistant.agent.permissions import PermissionEngine
from codebase_assistant.agent.policy import PolicyEngine
from codebase_assistant.agent.repair import RepairWorkflowCoordinator
from codebase_assistant.agent.repair_models import RepairWorkflowStatus
from codebase_assistant.agent.tools import ToolRegistry
from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.generation.code_generator import CodeProposalGenerator
from codebase_assistant.generation.test_generator import TestProposalGenerator
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.llm.mock_client import MockGeminiClient
from codebase_assistant.retrieval.hybrid import HybridRetriever


class TestRepairWorkflowCoordinator(unittest.TestCase):
    """Test suite verifying end-to-end planning, patch application, diagnostics, and bounded repair."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.tmp_dir.name)

        # Create source files
        src_dir = self.repo_dir / "src"
        src_dir.mkdir(parents=True)
        self.calc_file = src_dir / "calc.py"
        self.calc_file.write_text(
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n\n"
            "def div(a: int, b: int) -> float:\n"
            "    return a / b\n",
            encoding="utf-8",
        )

        tests_dir = self.repo_dir / "tests" / "unit"
        tests_dir.mkdir(parents=True)
        self.test_calc_file = tests_dir / "test_calc.py"
        self.test_calc_file.write_text(
            "import unittest\n"
            "from src.calc import add, div\n\n"
            "class TestCalc(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n\n"
            "    def test_div(self):\n"
            "        self.assertEqual(div(10, 2), 5.0)\n",
            encoding="utf-8",
        )

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

        self.coordinator = RepairWorkflowCoordinator(
            repository_path=self.repo_dir,
            retriever=self.retriever,
            code_generator=self.code_gen,
            test_generator=self.test_gen,
            gemini_client=self.mock_client,
            policy_engine=self.policy,
            permission_engine=self.perm_engine,
            max_retries=3,
            command_timeout_seconds=10.0,
        )

    def tearDown(self):
        self.tmp_dir.cleanup()
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_plan_creation(self):
        plan = self.coordinator.create_plan("Fix division by zero in src/calc.py")
        self.assertEqual(plan.task, "Fix division by zero in src/calc.py")
        self.assertIn("src/calc.py", plan.target_files)
        self.assertTrue(len(plan.steps) > 0)
        self.assertTrue(len(plan.success_criteria) > 0)

    def test_single_attempt_repair_success(self):
        # Initial run: add function and verify tests pass
        report = self.coordinator.run(
            task="Add helper function in src/calc.py",
            target_file="src/calc.py",
            test_command="python -m unittest tests/unit/test_calc.py",
        )

        self.assertEqual(report.status, RepairWorkflowStatus.SUCCESS)
        self.assertEqual(report.total_attempts, 1)
        self.assertIsNotNone(report.final_diff)
        self.assertIn("src/calc.py", report.modified_files)

        # Verify disk state contains changes
        current_content = self.calc_file.read_text(encoding="utf-8")
        self.assertIn("def proposed_enhancement", current_content)

    def test_multi_attempt_self_repair(self):
        # Update test to require zero division defense
        self.test_calc_file.write_text(
            "import unittest\n"
            "from src.calc import add, div\n\n"
            "class TestCalc(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n\n"
            "    def test_div_zero(self):\n"
            "        with self.assertRaises(ValueError):\n"
            "            div(10, 0)\n",
            encoding="utf-8",
        )

        # Queue responses:
        # 1. Plan response
        plan_json = (
            '{\n'
            '  "analysis": "Fix zero division in div",\n'
            '  "target_files": ["src/calc.py"],\n'
            '  "test_files": ["tests/unit/test_calc.py"],\n'
            '  "steps": ["Apply fix", "Run tests"],\n'
            '  "success_criteria": ["Tests pass"]\n'
            '}'
        )
        # 2. First proposal doesn't handle zero division (fails test)
        bad_proposal = (
            "--- ARCHITECTURAL REASONING ---\nAttempt 1\n\n"
            "--- FILE: src/calc.py ---\n"
            "```python\n"
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n\n"
            "def div(a: int, b: int) -> float:\n"
            "    return a / b\n"
            "```"
        )
        # 3. Second proposal fixes zero division (passes test)
        good_proposal = (
            "--- ARCHITECTURAL REASONING ---\nAttempt 2\n\n"
            "--- FILE: src/calc.py ---\n"
            "```python\n"
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n\n"
            "def div(a: int, b: int) -> float:\n"
            "    if b == 0:\n"
            "        raise ValueError('Cannot divide by zero')\n"
            "    return a / b\n"
            "```"
        )
        self.mock_client.responses_queue = [plan_json, bad_proposal, good_proposal]

        report = self.coordinator.run(
            task="Fix division by zero in src/calc.py",
            target_file="src/calc.py",
            test_command="python -m unittest tests/unit/test_calc.py",
        )

        self.assertEqual(report.status, RepairWorkflowStatus.SUCCESS)
        self.assertEqual(report.total_attempts, 2)
        self.assertFalse(report.attempts[0].test_passed)
        self.assertTrue(report.attempts[1].test_passed)
        self.assertIn("raise ValueError", self.calc_file.read_text(encoding="utf-8"))

    def test_max_retries_exceeded_rolls_back_workspace(self):
        calc_original = self.calc_file.read_text(encoding="utf-8")

        # Test requires an impossible assertion
        self.test_calc_file.write_text(
            "import unittest\n\n"
            "class TestCalc(unittest.TestCase):\n"
            "    def test_impossible(self):\n"
            "        self.assertEqual(1, 999)\n",
            encoding="utf-8",
        )

        self.coordinator.max_retries = 2
        report = self.coordinator.run(
            task="Attempt impossible task",
            target_file="src/calc.py",
            test_command="python -m unittest tests/unit/test_calc.py",
        )

        self.assertEqual(report.status, RepairWorkflowStatus.MAX_RETRIES_EXCEEDED)
        self.assertEqual(report.total_attempts, 2)
        self.assertIsNone(report.final_diff)

        # Workspace MUST be completely rolled back to original state
        calc_after = self.calc_file.read_text(encoding="utf-8")
        self.assertEqual(calc_original, calc_after)

    def test_user_denial_halts_and_rolls_back(self):
        calc_original = self.calc_file.read_text(encoding="utf-8")

        denying_engine = PermissionEngine(approval_callback=lambda req: False)
        self.coordinator.permission_engine = denying_engine
        self.coordinator.tool_registry.permission_engine = denying_engine

        report = self.coordinator.run(
            task="Add helper",
            target_file="src/calc.py",
            test_command="python -m unittest tests/unit/test_calc.py",
        )

        self.assertEqual(report.status, RepairWorkflowStatus.STOPPED_BY_USER)
        self.assertEqual(self.calc_file.read_text(encoding="utf-8"), calc_original)

    def test_target_discovery_from_task(self):
        report = self.coordinator.run(
            task="Update calc function",
            test_command="python -m unittest tests/unit/test_calc.py",
        )
        self.assertEqual(report.status, RepairWorkflowStatus.SUCCESS)
        self.assertIn("src/calc.py", report.plan.target_files)


if __name__ == "__main__":
    unittest.main()
