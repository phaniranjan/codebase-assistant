"""Unit tests for CodeTokenizer and BM25Retriever."""

import unittest

from codebase_assistant.ingestion.models import ChunkType, CodeChunk
from codebase_assistant.retrieval.bm25 import BM25Retriever
from codebase_assistant.retrieval.tokenizer import CodeTokenizer


class TestBM25Retriever(unittest.TestCase):
    """Test suite for CodeTokenizer and BM25Retriever."""

    def test_code_tokenizer_splits_camel_and_snake(self):
        tokens = CodeTokenizer.tokenize("class IncrementalIndexer(BaseIndexer): def parse_source_code(): pass")
        self.assertIn("incremental", tokens)
        self.assertIn("indexer", tokens)
        self.assertIn("parse", tokens)
        self.assertIn("source", tokens)
        self.assertIn("code", tokens)

    def test_bm25_exact_symbol_match_boost(self):
        chunks = [
            CodeChunk(
                chunk_id="chunk_target",
                file_path="src/parser.py",
                content="def parse_ast(code: str):\n    return ast.parse(code)",
                symbol_name="parse_ast",
                language="python",
                chunk_type=ChunkType.FUNCTION,
                start_line=1,
                end_line=2,
                content_hash="hash1",
            ),
            CodeChunk(
                chunk_id="chunk_other",
                file_path="src/utils.py",
                content="def helper():\n    # Note: calls parse_ast sometimes\n    pass",
                symbol_name="helper",
                language="python",
                chunk_type=ChunkType.FUNCTION,
                start_line=1,
                end_line=3,
                content_hash="hash2",
            ),
        ]

        retriever = BM25Retriever()
        retriever.index(chunks)

        results = retriever.search("parse_ast", top_k=2)
        self.assertEqual(len(results), 2)
        # chunk_target should rank first with significantly higher score due to symbol boost
        self.assertEqual(results[0][0], "chunk_target")
        self.assertGreater(results[0][1], results[1][1])

    def test_bm25_empty_query(self):
        retriever = BM25Retriever()
        retriever.index([])
        self.assertEqual(retriever.search("test"), [])
        self.assertEqual(retriever.search(""), [])


if __name__ == "__main__":
    unittest.main()
