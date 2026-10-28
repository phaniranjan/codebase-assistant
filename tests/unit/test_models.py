"""Unit tests for ingestion data models."""

import unittest
from codebase_assistant.ingestion.models import (
    ChunkType,
    CodeChunk,
    FileMetadata,
    IndexManifest,
    SourceRevision,
)


class TestModels(unittest.TestCase):
    """Test serialization, deserialization, and methods of core data models."""

    def test_chunk_type_enum(self):
        self.assertEqual(ChunkType.CLASS.value, "class")
        self.assertEqual(ChunkType.FUNCTION.value, "function")
        self.assertEqual(ChunkType.METHOD.value, "method")
        self.assertEqual(ChunkType.MODULE_HEADER.value, "module_header")
        self.assertEqual(ChunkType.DOC_SECTION.value, "doc_section")
        self.assertEqual(ChunkType.CONFIG_BLOCK.value, "config_block")

    def test_code_chunk_creation_and_summary(self):
        chunk = CodeChunk(
            chunk_id="chunk_123",
            file_path="src/utils.py",
            content="def add(a, b):\n    return a + b",
            language="python",
            chunk_type=ChunkType.FUNCTION,
            symbol_name="add",
            start_line=1,
            end_line=2,
            content_hash="abc456",
            source_revision="commit_789",
        )
        self.assertEqual(chunk.chunk_id, "chunk_123")
        self.assertEqual(chunk.summary(), "[FUNCTION] (add) src/utils.py:1-2")

        # Test Pydantic model serialization round-trip
        data = chunk.model_dump()
        reconstructed = CodeChunk.model_validate(data)
        self.assertEqual(reconstructed.chunk_id, chunk.chunk_id)
        self.assertEqual(reconstructed.content_hash, "abc456")

    def test_index_manifest_serialization(self):
        manifest = IndexManifest(
            repository_path="/workspace/repo",
            source_revision=SourceRevision(commit_sha="abcdef", branch="main", revision_id="abcdef"),
            indexed_at="2026-10-28T22:00:00Z",
            file_count=1,
            chunk_count=1,
            files={
                "src/utils.py": FileMetadata(
                    file_path="src/utils.py",
                    file_hash="hash1",
                    file_size=50,
                    last_modified=1700000000.0,
                    language="python",
                    chunk_ids=["chunk_123"],
                )
            },
        )
        data = manifest.model_dump()
        reconstructed = IndexManifest.model_validate(data)
        self.assertEqual(reconstructed.source_revision.commit_sha, "abcdef")
        self.assertIn("src/utils.py", reconstructed.files)


if __name__ == "__main__":
    unittest.main()
