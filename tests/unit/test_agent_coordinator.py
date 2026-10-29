"""Unit tests for AgentCoordinator and stopping condition boundaries."""

from pathlib import Path
import shutil
import tempfile
import unittest

from codebase_assistant.agent.coordinator import AgentCoordinator
from codebase_assistant.agent.models import AgentConfig, AgentStoppingReason
from codebase_assistant.agent.permissions import PermissionEngine
from codebase_assistant.agent.policy import PolicyEngine
from codebase_assistant.agent.tools import ToolRegistry
from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.llm.mock_client import MockGeminiClient
from codebase_assistant.retrieval.hybrid import HybridRetriever


class TestAgentCoordinator(unittest.TestCase):
    """Test suite verifying agent orchestration loop, safety bounds, and stopping conditions."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.tmp_dir.name)

        src_dir = self.repo_dir / "src"
        src_dir.mkdir(parents=True)
        (src_dir / "auth.py").write_text(
            "def authenticate(token: str) -> bool:\n    return token == 'secret'\n",
            encoding="utf-8",
        )

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

    def tearDown(self):
        self.tmp_dir.cleanup()
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_agent_completes_task_successfully(self):
        client = MockGeminiClient(
            responses_queue=[
                'Thought: I need to search the codebase for authentication logic.\nAction: search_codebase({"query": "authenticate"})',
                'Thought: I have the source chunk.\nFinal Answer: The authentication function is defined in src/auth.py.',
            ]
        )
        registry = ToolRegistry(
            repository_path=self.repo_dir,
            retriever=self.retriever,
            permission_engine=PermissionEngine(auto_approve_consequential=True),
        )
        coordinator = AgentCoordinator(tool_registry=registry, llm_client=client)

        resp = coordinator.run("Where is authenticate defined?")

        self.assertEqual(resp.stopping_reason, AgentStoppingReason.COMPLETED)
        self.assertEqual(resp.total_steps, 2)
        self.assertEqual(resp.total_tool_calls, 1)
        self.assertIn("src/auth.py", resp.final_answer)

    def test_agent_stops_at_max_iterations(self):
        # Client that calls tools up to limit, then provides final answer on iteration notice
        client = MockGeminiClient(
            responses_queue=[
                'Thought: Step 1.\nAction: search_codebase({"query": "q1"})',
                'Thought: Step 2.\nAction: search_codebase({"query": "q2"})',
                'Final Answer: Terminated after max iterations.',
            ]
        )
        registry = ToolRegistry(
            repository_path=self.repo_dir,
            retriever=self.retriever,
        )
        config = AgentConfig(max_iterations=2, max_reasoning_steps=10)
        coordinator = AgentCoordinator(tool_registry=registry, llm_client=client, config=config)

        resp = coordinator.run("Search repeatedly")

        self.assertEqual(resp.stopping_reason, AgentStoppingReason.MAX_ITERATIONS)
        self.assertEqual(resp.total_tool_calls, 2)
        self.assertIn("Terminated after max iterations", resp.final_answer)

    def test_agent_stops_at_max_reasoning_steps(self):
        # Client that generates thoughts without calling tools
        client = MockGeminiClient(
            responses_queue=[
                'Thought: Reasoning without action 1.',
                'Thought: Reasoning without action 2.',
                'Thought: Reasoning without action 3.',
            ]
        )
        registry = ToolRegistry(
            repository_path=self.repo_dir,
            retriever=self.retriever,
        )
        config = AgentConfig(max_iterations=5, max_reasoning_steps=2)
        coordinator = AgentCoordinator(tool_registry=registry, llm_client=client, config=config)

        resp = coordinator.run("Think hard")

        self.assertEqual(resp.stopping_reason, AgentStoppingReason.MAX_REASONING_STEPS)
        self.assertEqual(resp.total_steps, 2)
        self.assertEqual(resp.total_tool_calls, 0)

    def test_agent_stops_at_token_budget_exceeded(self):
        client = MockGeminiClient(
            responses_queue=[
                'Thought: Initial thought.\nAction: search_codebase({"query": "auth"})',
            ]
        )
        registry = ToolRegistry(
            repository_path=self.repo_dir,
            retriever=self.retriever,
        )
        # Extremely low token budget so prompt exceeds it
        config = AgentConfig(token_budget=10)
        coordinator = AgentCoordinator(tool_registry=registry, llm_client=client, config=config)

        resp = coordinator.run("Test budget limit")

        self.assertEqual(resp.stopping_reason, AgentStoppingReason.BUDGET_EXCEEDED)
        self.assertIn("budget limit", resp.final_answer)

    def test_agent_stops_when_authorization_denied(self):
        client = MockGeminiClient(
            responses_queue=[
                'Thought: I will apply this patch.\nAction: apply_patch({"unified_diff": "--- a/src/auth.py\\n+++ b/src/auth.py\\n@@ -1,1 +1,1 @@\\n-x\\n+y\\n"})',
            ]
        )
        # Denying permission engine
        denying_engine = PermissionEngine(approval_callback=lambda req: False)
        registry = ToolRegistry(
            repository_path=self.repo_dir,
            retriever=self.retriever,
            permission_engine=denying_engine,
        )
        coordinator = AgentCoordinator(tool_registry=registry, llm_client=client)

        resp = coordinator.run("Apply patch to auth.py")

        self.assertEqual(resp.stopping_reason, AgentStoppingReason.AUTHORIZATION_DENIED)
        self.assertIn("Authorization was denied", resp.final_answer)


if __name__ == "__main__":
    unittest.main()
