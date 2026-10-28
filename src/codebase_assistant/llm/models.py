"""Data models for LLM interactions, citations, and repository question answering."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Source reference citation pointing to an exact file span and symbol."""

    file_path: str
    start_line: int
    end_line: int
    symbol_name: Optional[str] = None
    snippet: Optional[str] = None

    def format_citation(self) -> str:
        """Return human-readable markdown citation string."""
        sym = f" (`{self.symbol_name}`)" if self.symbol_name else ""
        return f"`{self.file_path}` (lines {self.start_line}-{self.end_line}{sym})"


class LLMResponse(BaseModel):
    """Standardized response from an LLM provider."""

    text: str
    model_name: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None


class QAResponse(BaseModel):
    """Final answer returned by the Codebase Q&A engine with citations and retrieval metadata."""

    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunks_count: int = 0
    assembled_chunks_count: int = 0
    omitted_chunks_count: int = 0
    model_used: str = ""
    is_truncated: bool = False
    context_tokens: int = 0
