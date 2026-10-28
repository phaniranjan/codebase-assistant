"""Unit tests for ContextAssembler."""

import unittest

from codebase_assistant.ingestion.models import ChunkType, CodeChunk
from codebase_assistant.retrieval.context_assembler import ContextAssembler
from codebase_assistant.retrieval.hybrid import SearchResult


class TestContextAssembler(unittest.TestCase):
    """Test suite for ContextAssembler."""

    def setUp(self):
        self.chunk1 = CodeChunk(
            chunk_id="c1",
            file_path="src/main.py",
            content="def start():\n    print('start')\n",
            symbol_name="start",
            language="python",
            chunk_type=ChunkType.FUNCTION,
            start_line=1,
            end_line=2,
            content_hash="h1",
        )
        self.chunk2 = CodeChunk(
            chunk_id="c2",
            file_path="src/config.py",
            content="DEBUG = True\nPORT = 8080\n" * 50,  # ~1300 chars
            symbol_name="config",
            language="python",
            chunk_type=ChunkType.CONFIG_BLOCK,
            start_line=1,
            end_line=100,
            content_hash="h2",
        )

        self.results = [
            SearchResult(chunk_id="c1", chunk=self.chunk1, score=0.9),
            SearchResult(chunk_id="c2", chunk=self.chunk2, score=0.7),
        ]

    def test_context_assembler_formatting(self):
        assembler = ContextAssembler(default_max_tokens=1000)
        context = assembler.assemble(self.results)

        self.assertIn("### [Reference 1] `src/main.py` (lines 1-2, symbol: `start`)", context.formatted_text)
        self.assertIn("def start():", context.formatted_text)
        self.assertEqual(len(context.included_chunks), 2)
        self.assertEqual(context.omitted_chunks_count, 0)
        self.assertFalse(context.is_truncated())

    def test_token_budget_truncation(self):
        # Small budget: 50 tokens (approx 200 characters)
        # chunk1 is ~100 chars, chunk2 is ~1300 chars, so chunk2 should be omitted
        assembler = ContextAssembler(default_max_tokens=50)
        context = assembler.assemble(self.results, max_tokens=50)

        self.assertEqual(len(context.included_chunks), 1)
        self.assertEqual(context.omitted_chunks_count, 1)
        self.assertTrue(context.is_truncated())
        self.assertEqual(context.included_chunks[0].chunk_id, "c1")

    def test_deduplication_of_identical_spans(self):
        assembler = ContextAssembler()
        duplicate_results = [
            SearchResult(chunk_id="c1", chunk=self.chunk1, score=0.9),
            SearchResult(chunk_id="c1_dup", chunk=self.chunk1, score=0.85),
        ]
        context = assembler.assemble(duplicate_results)
        self.assertEqual(len(context.included_chunks), 1)

    def test_first_chunk_exceeds_budget_is_omitted(self):
        # A tiny budget of 5 tokens (~17 characters) cannot fit chunk1 (~100 chars)
        assembler = ContextAssembler(default_max_tokens=5)
        context = assembler.assemble(self.results, max_tokens=5)

        self.assertEqual(len(context.included_chunks), 0)
        self.assertEqual(context.omitted_chunks_count, 2)
        self.assertTrue(context.is_truncated())
        self.assertEqual(context.formatted_text, "")


if __name__ == "__main__":
    unittest.main()
