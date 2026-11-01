"""Repository ingestion, scanning, AST parsing, and logical chunking."""

from codebase_assistant.ingestion.chunker import LogicalChunker
from codebase_assistant.ingestion.indexer import IncrementalIndexer
from codebase_assistant.ingestion.models import (
    ChunkType,
    CodeChunk,
    FileMetadata,
    IndexManifest,
    SourceRevision,
)
from codebase_assistant.ingestion.parser import ConfigParser, MarkdownParser, PythonASTParser
from codebase_assistant.ingestion.scanner import RepositoryScanner

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
