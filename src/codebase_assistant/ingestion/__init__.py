"""Repository ingestion, scanning, AST parsing, and logical chunking."""

from codebase_assistant.ingestion.models import (
    ChunkType,
    CodeChunk,
    FileMetadata,
    IndexManifest,
    SourceRevision,
)
from codebase_assistant.ingestion.scanner import RepositoryScanner
from codebase_assistant.ingestion.parser import PythonASTParser, MarkdownParser, ConfigParser
from codebase_assistant.ingestion.chunker import LogicalChunker
from codebase_assistant.ingestion.indexer import IncrementalIndexer

__all__ = [
    "ChunkType",
    "CodeChunk",
    "FileMetadata",
    "IndexManifest",
    "SourceRevision",
    "RepositoryScanner",
    "PythonASTParser",
    "MarkdownParser",
    "ConfigParser",
    "LogicalChunker",
    "IncrementalIndexer",
]
