"""Integration test verifying code and test proposal generation against read-only ai-evaluation."""

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from click.testing import CliRunner

from codebase_assistant.cli import main
from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.generation.code_generator import CodeProposalGenerator
from codebase_assistant.generation.diff_engine import DiffEngine
from codebase_assistant.generation.models import ProposalType
from codebase_assistant.generation.test_generator import TestProposalGenerator
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.llm.mock_client import MockGeminiClient
from codebase_assistant.retrieval.hybrid import HybridRetriever


class TestAiEvalProposalsIntegration(unittest.TestCase):
    """End-to-end integration tests for code & test proposal generation on ai-evaluation."""

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

            # Pre-index and generate mock embeddings once for integration tests
            indexer = IncrementalIndexer(cls.ai_eval_dir, cache_dir=cls.cache_dir)
            indexer.index()

            cls.emb_provider = MockEmbeddingProvider()
            manager = EmbeddingManager(cls.ai_eval_dir, cache_dir=cls.cache_dir, provider=cls.emb_provider)
            manager.sync()

            chunks = indexer.load_chunks()
            vector_store = manager.load_vector_store()
            cls.retriever = HybridRetriever(
                chunks=chunks,
                vector_store=vector_store,
                embedding_provider=cls.emb_provider,
            )
            cls.llm_client = MockGeminiClient()

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

        # Strict cryptographic SHA-256 snapshot of all target repository files before each test
        self.initial_hashes = self._snapshot_target_tree(self.ai_eval_dir)

    def tearDown(self):
        # Strict cryptographic verification that target repository was 100% read-only across all files
        current_hashes = self._snapshot_target_tree(self.ai_eval_dir)
        self.assertEqual(
            self.initial_hashes,
            current_hashes,
            "Target repository was modified during proposal generation! Cryptographic integrity violation.",
        )

    def test_ai_eval_code_proposal_generation(self):
        generator = CodeProposalGenerator(
            repository_path=self.ai_eval_dir,
            retriever=self.retriever,
            llm_client=self.llm_client,
        )

        target_file = "src/ai_evaluation/rag_pipeline.py"
        result = generator.propose(
            task="Add execution latency tracking to rag pipeline query",
            target_file=target_file,
            top_k=3,
        )

        self.assertTrue(result.is_valid, f"Proposal verification failed: {result.verification_summary}")
        self.assertEqual(len(result.proposals), 1)

        prop = result.proposals[0]
        self.assertEqual(prop.target_file, target_file)
        self.assertEqual(prop.proposal_type, ProposalType.MODIFICATION)
        self.assertTrue(prop.ast_valid, f"AST syntax errors: {prop.syntax_errors}")
        self.assertIn("--- a/src/ai_evaluation/rag_pipeline.py", prop.unified_diff)
        self.assertIn("+++ b/src/ai_evaluation/rag_pipeline.py", prop.unified_diff)

        # Verify in-memory patch application and AST verification
        patched = DiffEngine.apply_diff_in_memory(prop.original_content or "", prop.unified_diff)
        self.assertEqual(patched, prop.proposed_content)
        is_valid, errs = DiffEngine.validate_ast_syntax(patched)
        self.assertTrue(is_valid, f"AST errors in patched code: {errs}")

    def test_ai_eval_test_proposal_generation(self):
        generator = TestProposalGenerator(
            repository_path=self.ai_eval_dir,
            retriever=self.retriever,
            llm_client=self.llm_client,
        )

        source_file = "src/ai_evaluation/rag_retriever.py"
        result = generator.propose_tests(
            source_file=source_file,
            top_k=3,
        )

        self.assertTrue(result.is_valid, f"Test proposal verification failed: {result.verification_summary}")
        self.assertEqual(len(result.proposals), 1)

        prop = result.proposals[0]
        self.assertEqual(prop.proposal_type, ProposalType.TEST_PROPOSAL)
        self.assertTrue(prop.target_file.startswith("tests/"))
        self.assertTrue(prop.ast_valid, f"AST syntax errors: {prop.syntax_errors}")
        self.assertIn(f"+++ b/{prop.target_file}", prop.unified_diff)

        # Verify in-memory patch application
        patched = DiffEngine.apply_diff_in_memory(prop.original_content or "", prop.unified_diff)
        self.assertEqual(patched, prop.proposed_content)
        is_valid, errs = DiffEngine.validate_ast_syntax(patched)
        self.assertTrue(is_valid, f"AST errors in test proposal: {errs}")

    def test_ai_eval_cli_proposals_end_to_end(self):
        runner = CliRunner()
        external_patch_dir = Path(tempfile.mkdtemp())
        code_patch = external_patch_dir / "code_change.patch"
        test_patch = external_patch_dir / "test_suite.patch"

        try:
            # 1. Test propose-code CLI
            res_code = runner.invoke(
                main,
                [
                    "propose-code",
                    "Add timeout configuration to rag pipeline",
                    str(self.ai_eval_dir),
                    "--file",
                    "src/ai_evaluation/rag_pipeline.py",
                    "--cache-dir",
                    str(self.cache_dir),
                    "--provider",
                    "mock",
                    "--output-patch",
                    str(code_patch),
                ],
            )
            self.assertEqual(res_code.exit_code, 0, res_code.output)
            self.assertTrue(code_patch.is_file())
            self.assertIn("AST Syntax: Valid Python", res_code.output)
            self.assertIn("Saved unified diff patch to:", res_code.output)

            # 2. Test propose-tests CLI
            res_test = runner.invoke(
                main,
                [
                    "propose-tests",
                    "src/ai_evaluation/rag_retriever.py",
                    str(self.ai_eval_dir),
                    "--cache-dir",
                    str(self.cache_dir),
                    "--provider",
                    "mock",
                    "--output-patch",
                    str(test_patch),
                ],
            )
            self.assertEqual(res_test.exit_code, 0, res_test.output)
            self.assertTrue(test_patch.is_file())
            self.assertIn("AST Syntax: Valid Python", res_test.output)
            self.assertIn("Saved unified diff patch to:", res_test.output)
        finally:
            shutil.rmtree(external_patch_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
