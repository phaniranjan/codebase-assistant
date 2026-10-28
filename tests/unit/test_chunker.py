"""Unit tests for logical chunker."""

import unittest

from codebase_assistant.ingestion.chunker import LogicalChunker
from codebase_assistant.ingestion.models import ChunkType

SAMPLE_CODE = '''"""Sample module."""


def calculate_metrics(values: list) -> dict:
    """Calculate mean and total."""
    total = sum(values)
    mean = total / len(values) if values else 0.0
    return {"total": total, "mean": mean}
'''


class TestLogicalChunker(unittest.TestCase):
    """Test chunk generation, hashing, and subdivision."""

    def setUp(self):
        self.chunker = LogicalChunker(max_chunk_chars=500)

    def test_chunk_file_creates_valid_code_chunks(self):
        chunks = self.chunker.chunk_file(
            content=SAMPLE_CODE,
            rel_file_path="src/metrics.py",
            language="python",
            source_revision="sha_123",
        )

        self.assertTrue(len(chunks) >= 1)
        func_chunk = next(c for c in chunks if c.symbol_name == "calculate_metrics")

        self.assertEqual(func_chunk.chunk_type, ChunkType.FUNCTION)
        self.assertEqual(func_chunk.file_path, "src/metrics.py")
        self.assertEqual(func_chunk.source_revision, "sha_123")
        self.assertEqual(func_chunk.docstring, "Calculate mean and total.")
        self.assertTrue(len(func_chunk.content_hash) == 64)  # SHA-256
        self.assertIn("calculate_metrics", func_chunk.chunk_id)

    def test_chunk_subdivision_for_large_symbols(self):
        # Create a large symbol exceeding max_chunk_chars (500 chars)
        large_lines = [f"    # Line {i} with some descriptive comments filling space" for i in range(50)]
        large_code = "def large_handler():\n    '''Handler docstring.'''\n" + "\n".join(large_lines) + "\n    return True\n"

        small_chunker = LogicalChunker(max_chunk_chars=300, chunk_overlap_lines=5)
        chunks = small_chunker.chunk_file(
            content=large_code,
            rel_file_path="src/large.py",
            language="python",
            source_revision="sha_456",
        )

        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        self.assertTrue(len(func_chunks) > 1)
        self.assertTrue(all("large_handler_part" in c.symbol_name for c in func_chunks))
        self.assertTrue(all(c.start_line > 0 and c.end_line >= c.start_line for c in func_chunks))


if __name__ == "__main__":
    unittest.main()
