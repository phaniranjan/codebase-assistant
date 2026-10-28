"""Data models for repository ingestion, code chunking, and index manifest."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    """Categorization of logical chunks extracted from source code or documentation."""

    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    MODULE_HEADER = "module_header"
    DOC_SECTION = "doc_section"
    CONFIG_BLOCK = "config_block"
    CODE_BLOCK = "code_block"
    UNKNOWN = "unknown"


class SourceRevision(BaseModel):
    """Source control revision information for freshness and stale detection."""

    commit_sha: Optional[str] = None
    branch: Optional[str] = None
    is_dirty: bool = False
    revision_id: str = "unknown"


class CodeChunk(BaseModel):
    """A logically segmented chunk of code or documentation with rich metadata."""

    chunk_id: str
    file_path: str  # Repository-relative path (e.g. src/ai_evaluation/rag_pipeline.py)
    content: str
    language: str  # python, markdown, json, toml, etc.
    chunk_type: ChunkType
    symbol_name: Optional[str] = None  # e.g., BM25Retriever._calc_idf
    parent_symbol: Optional[str] = None  # e.g., BM25Retriever
    start_line: int
    end_line: int
    content_hash: str  # SHA-256 of chunk content
    source_revision: Optional[str] = None  # Commit SHA or revision marker
    docstring: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)  # imported modules / referenced symbols
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> str:
        """Brief descriptive summary for logging and retrieval display."""
        symbol = f" ({self.symbol_name})" if self.symbol_name else ""
        return f"[{self.chunk_type.value.upper()}]{symbol} {self.file_path}:{self.start_line}-{self.end_line}"


class FileMetadata(BaseModel):
    """Metadata tracking per indexed file for incremental indexing."""

    file_path: str
    file_hash: str  # SHA-256 of complete file
    file_size: int
    last_modified: float
    language: str
    source_revision: Optional[str] = None
    chunk_ids: List[str] = Field(default_factory=list)


class IndexManifest(BaseModel):
    """Top-level manifest of the repository knowledge base."""

    repository_path: str
    source_revision: Optional[SourceRevision] = None
    indexed_at: str
    file_count: int = 0
    chunk_count: int = 0
    files: Dict[str, FileMetadata] = Field(default_factory=dict)
    chunks: Dict[str, CodeChunk] = Field(default_factory=dict)
