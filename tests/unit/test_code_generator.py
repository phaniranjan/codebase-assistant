"""Unit tests for CodeProposalGenerator."""

from pathlib import Path
import tempfile
import unittest

from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.generation.code_generator import CodeProposalGenerator
from codebase_assistant.generation.models import ProposalType
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.llm.mock_client import MockGeminiClient
from codebase_assistant.retrieval.hybrid import HybridRetriever


class TestCodeProposalGenerator(unittest.TestCase):
    """Test suite verifying code proposal generation and AST validation."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.tmp_dir.name)

        # Create sample codebase
        src_dir = self.repo_dir / "src"
        src_dir.mkdir(parents=True)
        self.calc_file = src_dir / "calc.py"
        self.calc_content = (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n\n"
            "def divide(a: int, b: int) -> float:\n"
            "    return a / b\n"
        )
        self.calc_file.write_text(self.calc_content, encoding="utf-8")

        # Index and create retriever
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
        self.mock_client = MockGeminiClient(model_name="mock-gemini-v1")

    def tearDown(self):
        self.tmp_dir.cleanup()
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_propose_code_modification_with_ast_validation(self):
        generator = CodeProposalGenerator(
            repository_path=self.repo_dir,
            retriever=self.retriever,
            llm_client=self.mock_client,
        )

        result = generator.propose(
            task="Add zero-division handling to divide function",
            target_file="src/calc.py",
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.proposals), 1)
        prop = result.proposals[0]

        self.assertEqual(prop.target_file, "src/calc.py")
        self.assertEqual(prop.proposal_type, ProposalType.MODIFICATION)
        self.assertFalse(prop.is_new_file)
        self.assertTrue(prop.ast_valid)
        self.assertEqual(len(prop.syntax_errors), 0)

        # Verify unified diff format
        self.assertIn("--- a/src/calc.py", prop.unified_diff)
        self.assertIn("+++ b/src/calc.py", prop.unified_diff)
        self.assertTrue(len(prop.architectural_reasoning) > 0)
        self.assertIn("proposed modification implements", prop.architectural_reasoning)

        # Verify target repository was NOT modified
        disk_content = self.calc_file.read_text(encoding="utf-8")
        self.assertEqual(disk_content, self.calc_content, "Target file was directly modified on disk!")

    def test_propose_new_file(self):
        generator = CodeProposalGenerator(
            repository_path=self.repo_dir,
            retriever=self.retriever,
            llm_client=self.mock_client,
        )

        result = generator.propose(
            task="Create a new math utilities module",
            target_file="src/utils.py",
        )

        self.assertTrue(result.is_valid)
        prop = result.proposals[0]
        self.assertEqual(prop.target_file, "src/utils.py")
        self.assertEqual(prop.proposal_type, ProposalType.NEW_FILE)
        self.assertTrue(prop.is_new_file)
        self.assertTrue(prop.ast_valid)
        self.assertIn("--- /dev/null", prop.unified_diff)
        self.assertIn("+++ b/src/utils.py", prop.unified_diff)

        # File must not exist on disk
        self.assertFalse((self.repo_dir / "src/utils.py").exists())

    def test_propose_rejects_path_traversal(self):
        generator = CodeProposalGenerator(
            repository_path=self.repo_dir,
            retriever=self.retriever,
            llm_client=self.mock_client,
        )

        result = generator.propose(
            task="Malicious file write",
            target_file="../../etc/passwd",
        )

        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.proposals), 0)
        self.assertIn("Path security violation", result.verification_summary)


if __name__ == "__main__":
    unittest.main()
