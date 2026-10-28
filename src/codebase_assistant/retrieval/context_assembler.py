"""Context assembler formatting retrieved chunks with strict token budgeting and citations."""

import math
from typing import List

from pydantic import BaseModel, Field

from codebase_assistant.retrieval.hybrid import SearchResult


class AssembledContext(BaseModel):
    """Result of context assembly, including prompt-ready text and budgeting metrics."""

    formatted_text: str
    included_chunks: List[SearchResult] = Field(default_factory=list)
    omitted_chunks_count: int = 0
    estimated_tokens: int = 0
    max_tokens: int = 8192

    def is_truncated(self) -> bool:
        return self.omitted_chunks_count > 0


class ContextAssembler:
    """Formats retrieved code chunks for LLM prompts with token budgeting and citation headers.

    Uses a conservative estimation of ~3.5 characters per token for source code to ensure
    the prompt payload never overflows Gemini's token window. Strictly enforces the budget
    on all chunks (including the initial chunk).
    """

    CHARS_PER_TOKEN = 3.5

    def __init__(self, default_max_tokens: int = 8192) -> None:
        self.default_max_tokens = default_max_tokens

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """Estimate token count conservatively for code and natural language text."""
        if not text:
            return 0
        return max(1, int(math.ceil(len(text) / cls.CHARS_PER_TOKEN)))

    def assemble(
        self,
        results: List[SearchResult],
        max_tokens: int | None = None,
    ) -> AssembledContext:
        """Assemble retrieved search results into a formatted context block within token limits."""
        budget = max_tokens or self.default_max_tokens

        included: List[SearchResult] = []
        omitted_count = 0
        blocks: List[str] = []
        accumulated_chars = 0
        budget_chars = int(budget * self.CHARS_PER_TOKEN)

        seen_spans = set()  # (file_path, start_line, end_line)

        for rank, res in enumerate(results, start=1):
            chunk = res.chunk
            span_key = (chunk.file_path, chunk.start_line, chunk.end_line)
            if span_key in seen_spans:
                continue

            # Format citation header
            sym_desc = f", symbol: `{chunk.symbol_name}`" if chunk.symbol_name else ""
            header = (
                f"### [Reference {rank}] `{chunk.file_path}` "
                f"(lines {chunk.start_line}-{chunk.end_line}{sym_desc})\n"
            )
            lang = chunk.language or ""
            code_block = f"```{lang}\n{chunk.content}\n```\n"

            formatted_chunk = f"{header}{code_block}\n"
            chunk_chars = len(formatted_chunk)

            # Strict budget enforcement: applies to all candidates including the first chunk
            if accumulated_chars + chunk_chars > budget_chars:
                omitted_count += 1
                continue

            blocks.append(formatted_chunk)
            accumulated_chars += chunk_chars
            included.append(res)
            seen_spans.add(span_key)

        joined_text = "\n".join(blocks).strip()
        estimated = self.estimate_tokens(joined_text)

        return AssembledContext(
            formatted_text=joined_text,
            included_chunks=included,
            omitted_chunks_count=omitted_count,
            estimated_tokens=estimated,
            max_tokens=budget,
        )
