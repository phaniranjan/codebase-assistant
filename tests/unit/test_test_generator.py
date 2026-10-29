"""Unit tests for TestProposalGenerator."""

from pathlib import Path
import tempfile
import unittest

from codebase_assistant.embeddings.manager import EmbeddingManager
from codebase_assistant.embeddings.mock import MockEmbeddingProvider
from codebase_assistant.generation.models import ProposalType
from codebase_assistant.generation.test_generator import TestProposalGenerator
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.llm.mock_client import MockGeminiClient
from codebase_assistant.retrieval.hybrid import HybridRetriever


class TestTestProposalGenerator(unittest.TestCase):
    """Test suite verifying unit test proposal generation and AST validation."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.tmp_dir.name)

        # Create sample codebase with source and test layout
        src_dir = self.repo_dir / "src"
        src_dir.mkdir(parents=True)
        self.auth_file = src_dir / "auth.py"
        self.auth_content = (
            "def login(user: str, token: str) -> bool:\n"
            "    if not user or not token:\n"
            "        raise ValueError('Missing credentials')\n"
            "    return token == 'valid_token'\n"
        )
        self.auth_file.write_text(self.auth_content, encoding="utf-8")

        tests_dir = self.repo_dir / "tests" / "unit"
        tests_dir.mkdir(parents=True)
        (tests_dir / "test_sample.py").write_text("import unittest\n", encoding="utf-8")

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

    def test_framework_detection_and_target_file(self):
        generator = TestProposalGenerator(
            repository_path=self.repo_dir,
            retriever=self.retriever,
            llm_client=self.mock_client,
        )
        framework = generator._detect_testing_framework()
        self.assertEqual(framework, "unittest")

        target_test_file = generator._determine_target_test_file("src/auth.py")
        self.assertEqual(target_test_file, "tests/unit/test_auth.py")

    def test_propose_tests_for_source_file(self):
        generator = TestProposalGenerator(
            repository_path=self.repo_dir,
            retriever=self.retriever,
            llm_client=self.mock_client,
        )

        result = generator.propose_tests(source_file="src/auth.py")

        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.proposals), 1)
        prop = result.proposals[0]

        self.assertEqual(prop.target_file, "tests/unit/test_auth.py")
        self.assertEqual(prop.proposal_type, ProposalType.TEST_PROPOSAL)
        self.assertTrue(prop.ast_valid)
        self.assertGreaterEqual(len(prop.affected_symbols), 1)

        # Verify unified diff headers
        self.assertIn("--- /dev/null", prop.unified_diff)
        self.assertIn("+++ b/tests/unit/test_auth.py", prop.unified_diff)

        # Verify target repository was NOT modified on disk
        self.assertFalse((self.repo_dir / "tests/unit/test_auth.py").exists())


if __name__ == "__main__":
    unittest.main()
