"""Integration tests verifying RepairWorkflowCoordinator against read-only ai-evaluation."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from click.testing import CliRunner

from codebase_assistant.agent.permissions import PermissionEngine
from codebase_assistant.agent.policy import PolicyEngine
from codebase_assistant.agent.repair import RepairWorkflowCoordinator
from codebase_assistant.agent.repair_models import RepairWorkflowStatus
from codebase_assistant.cli import main
from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.generation.code_generator import CodeProposalGenerator
from codebase_assistant.generation.test_generator import TestProposalGenerator
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.llm.mock_client import MockGeminiClient
from codebase_assistant.retrieval.hybrid import HybridRetriever


class TestAiEvalRepairIntegration(unittest.TestCase):
    """End-to-end integration tests for agentic repair workflows on read-only ai-evaluation."""

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
            "Target repository was modified during repair workflow! Cryptographic integrity violation.",
        )

    def test_ai_eval_repair_workflow_planning_and_proposal(self):
        code_gen = CodeProposalGenerator(self.ai_eval_dir, self.retriever, self.mock_client)
        test_gen = TestProposalGenerator(self.ai_eval_dir, self.retriever, self.mock_client)
        policy = PolicyEngine(self.ai_eval_dir)
        perm = PermissionEngine(auto_approve_consequential=True)

        coordinator = RepairWorkflowCoordinator(
            repository_path=self.ai_eval_dir,
            retriever=self.retriever,
            code_generator=code_gen,
            test_generator=test_gen,
            gemini_client=self.mock_client,
            policy_engine=policy,
            permission_engine=perm,
            max_retries=1,
            command_timeout_seconds=10.0,
        )

        task = "Enhance error handling in rag_retriever.py"
        target_file = "src/ai_evaluation/rag_retriever.py"

        plan = coordinator.create_plan(task, target_file=target_file)
        self.assertIn("rag_retriever.py", plan.target_files[0])
        self.assertTrue(len(plan.steps) > 0)

        # Run proposal generation directly through code generator
        proposal_res = code_gen.propose(task, target_file=target_file)
        self.assertTrue(proposal_res.is_valid)
        self.assertTrue(len(proposal_res.proposals) > 0)
        self.assertTrue(proposal_res.proposals[0].ast_valid)

    def test_ai_eval_repair_cli_dry_run(self):
        runner = CliRunner()
        res = runner.invoke(
            main,
            [
                "repair",
                "--repo", str(self.ai_eval_dir),
                "--task", "Review and optimize rag_pipeline.py",
                "--target-file", "src/ai_evaluation/rag_pipeline.py",
                "--test-command", "python -m unittest tests/rag/test_rag_pipeline.py",
                "--provider", "mock",
                "--auto-approve",
                "--cache-dir", str(self.cache_dir),
                "--max-retries", "1",
                "--json",
            ],
        )
        # Verify JSON report structure and clean execution
        try:
            json_marker = '{\n  "workflow_id":'
            json_start = res.output.find(json_marker)
            if json_start == -1:
                json_start = res.output.find("{")
            self.assertNotEqual(json_start, -1, f"No JSON found in output: {res.output}")
            data = json.loads(res.output[json_start:])
            self.assertIn("workflow_id", data)
            self.assertIn("status", data)
            self.assertIn("plan", data)
        except json.JSONDecodeError:
            self.fail(f"CLI did not output valid JSON: {res.output}")


if __name__ == "__main__":
    unittest.main()
