"""Integration tests verifying AgentCoordinator and ToolRegistry against read-only ai-evaluation."""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from click.testing import CliRunner

from codebase_assistant.agent.coordinator import AgentCoordinator
from codebase_assistant.agent.models import AgentConfig, AgentStoppingReason
from codebase_assistant.agent.permissions import PermissionEngine
from codebase_assistant.agent.policy import PolicyEngine
from codebase_assistant.agent.tools import ToolRegistry
from codebase_assistant.cli import main
from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.generation.code_generator import CodeProposalGenerator
from codebase_assistant.generation.test_generator import TestProposalGenerator
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.llm.mock_client import MockGeminiClient
from codebase_assistant.retrieval.hybrid import HybridRetriever


class TestAiEvalAgentIntegration(unittest.TestCase):
    """End-to-end integration tests for autonomous agent execution on read-only ai-evaluation."""

    @classmethod
    def setUpClass(cls):
        candidates = [
            Path("../ai_evaluation").resolve(),
            Path("../../ai_evaluation").resolve(),
            Path("/home/phaniranjan/AI/ai_evaluation").resolve(),
            Path(r"\\wsl.localhost\Ubuntu-26.04\home\phaniranjan\AI\ai_evaluation").resolve(),
        ]
        cls.ai_eval_dir = None
        for cand in candidates:
            if cand.is_dir() and (cand / "src" / "ai_evaluation").is_dir():
                cls.ai_eval_dir = cand
                break

        if cls.ai_eval_dir is not None:
            # Allocate isolated temporary cache directory outside target repository
            cls.temp_cache = tempfile.TemporaryDirectory()
            cls.cache_dir = Path(cls.temp_cache.name)

            indexer = IncrementalIndexer(cls.ai_eval_dir, cache_dir=cls.cache_dir)
            indexer.index()
            chunks = indexer.load_chunks()

            cls.emb_provider = MockEmbeddingProvider()
            manager = EmbeddingManager(cls.ai_eval_dir, cache_dir=cls.cache_dir, provider=cls.emb_provider)
            manager.sync()
            vector_store = manager.load_vector_store()

            cls.retriever = HybridRetriever(
                chunks=chunks,
                vector_store=vector_store,
                embedding_provider=cls.emb_provider,
            )
            cls.mock_client = MockGeminiClient()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "temp_cache"):
            cls.temp_cache.cleanup()

    def _snapshot_target_tree(self, target_dir: Path) -> dict[str, str]:
        """Cryptographically hash every single file in the target repository tree."""
        hashes = {}
        ignored_dirs = {"deepeval_venv", ".venv", "venv", "env", ".git"}
        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            for f in files:
                p = Path(root) / f
                rel = p.relative_to(target_dir).as_posix()
                try:
                    content = p.read_bytes()
                    hashes[rel] = hashlib.sha256(content).hexdigest()
                except (PermissionError, OSError):
                    pass
        return hashes

    def setUp(self):
        if self.ai_eval_dir is None:
            self.skipTest("ai-evaluation repository not found on local path.")

        # Strict cryptographic snapshot of all target repository files before each test
        self.initial_hashes = self._snapshot_target_tree(self.ai_eval_dir)

    def tearDown(self):
        # Strict cryptographic verification that target repository was 100% read-only across all files
        current_hashes = self._snapshot_target_tree(self.ai_eval_dir)
        self.assertEqual(
            self.initial_hashes,
            current_hashes,
            "Target repository was modified during agent execution! Cryptographic integrity violation.",
        )

    def test_ai_eval_agent_search_and_read_trajectory(self):
        client = MockGeminiClient(
            responses_queue=[
                'Thought: I need to locate the RAG retriever implementation.\nAction: search_codebase({"query": "BM25Retriever"})',
                'Thought: I will inspect the source file directly.\nAction: read_source_file({"file_path": "src/ai_evaluation/rag_retriever.py", "start_line": 1, "end_line": 30})',
                'Thought: I have reviewed the implementation.\nFinal Answer: BM25Retriever is implemented in src/ai_evaluation/rag_retriever.py with tokenization and inverse document frequency scoring.',
            ]
        )
        policy = PolicyEngine(self.ai_eval_dir)
        perm = PermissionEngine(auto_approve_consequential=True)
        registry = ToolRegistry(
            repository_path=self.ai_eval_dir,
            retriever=self.retriever,
            policy_engine=policy,
            permission_engine=perm,
        )
        coordinator = AgentCoordinator(tool_registry=registry, llm_client=client)

        resp = coordinator.run("Explain how BM25Retriever is implemented in ai-evaluation")

        self.assertEqual(resp.stopping_reason, AgentStoppingReason.COMPLETED)
        self.assertEqual(resp.total_steps, 3)
        self.assertEqual(resp.total_tool_calls, 2)
        self.assertIn("BM25Retriever", resp.final_answer)

    def test_ai_eval_agent_code_proposal_generation(self):
        code_gen = CodeProposalGenerator(self.ai_eval_dir, self.retriever, self.mock_client)
        test_gen = TestProposalGenerator(self.ai_eval_dir, self.retriever, self.mock_client)
        policy = PolicyEngine(self.ai_eval_dir)
        perm = PermissionEngine(auto_approve_consequential=True)

        registry = ToolRegistry(
            repository_path=self.ai_eval_dir,
            retriever=self.retriever,
            code_generator=code_gen,
            test_generator=test_gen,
            policy_engine=policy,
            permission_engine=perm,
        )

        client = MockGeminiClient(
            responses_queue=[
                'Thought: I need to generate an enhancement proposal.\nAction: generate_code_proposal({"task": "Add latency metric", "target_file": "src/ai_evaluation/rag_pipeline.py"})',
                'Final Answer: Generated context-aware AST-verified unified diff proposal for rag_pipeline.py.',
            ]
        )
        coordinator = AgentCoordinator(tool_registry=registry, llm_client=client)

        resp = coordinator.run("Propose latency metrics in rag_pipeline.py")

        self.assertEqual(resp.stopping_reason, AgentStoppingReason.COMPLETED)
        self.assertEqual(len(resp.proposals), 1)
        prop = resp.proposals[0]["proposals"][0]
        self.assertTrue(prop["ast_valid"])
        self.assertIn("--- a/src/ai_evaluation/rag_pipeline.py", prop["unified_diff"])

    def test_ai_eval_agent_cli_end_to_end(self):
        runner = CliRunner()
        res = runner.invoke(
            main,
            [
                "agent",
                "Analyze rag retriever architecture",
                str(self.ai_eval_dir),
                "--cache-dir",
                str(self.cache_dir),
                "--provider",
                "mock",
                "--auto-approve",
            ],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Executing Agent Loop", res.output)
        self.assertIn("Stopping Reason: COMPLETED", res.output)
        self.assertIn("--- Final Answer ---", res.output)


if __name__ == "__main__":
    unittest.main()
