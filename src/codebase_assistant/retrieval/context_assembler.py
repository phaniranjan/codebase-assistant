"""Context assembler formatting retrieved chunks with strict token budgeting and citations."""

import math
from typing import Callable, List, Optional

from pydantic import BaseModel, Field

from codebase_assistant.retrieval.hybrid import SearchResult


class AssembledContext(BaseModel):
    """Result of context assembly, including prompt-ready text and budgeting metrics."""

    formatted_text: str
    full_prompt: str = ""
    included_chunks: List[SearchResult] = Field(default_factory=list)
    omitted_chunks_count: int = 0
    estimated_tokens: int = 0
    total_prompt_tokens: int = 0
    max_tokens: int = 8192

    def is_truncated(self) -> bool:
        return self.omitted_chunks_count > 0


class ContextAssembler:
    """Formats retrieved code chunks for LLM prompts with complete prompt budgeting and citation headers.

    Uses conservative token estimation (~3.5 characters per token) with support for model-specific
    token counters. The budget is allocated across the *entire* prompt payload:
      total_budget - system_instruction_tokens - user_query_tokens - overhead = chunk_budget
    ensuring that the final generated prompt never exceeds Gemini's context window.
    """

    CHARS_PER_TOKEN = 3.5

    def __init__(
        self,
        default_max_tokens: int = 8192,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        self.default_max_tokens = default_max_tokens
        self.token_counter = token_counter

    def count_tokens(self, text: str) -> int:
        """Count or conservatively estimate tokens for a given string."""
        if not text:
            return 0
        if self.token_counter is not None:
            try:
                return self.token_counter(text)
            except Exception:
                pass
        return max(1, int(math.ceil(len(text) / self.CHARS_PER_TOKEN)))

    def assemble(
        self,
        results: List[SearchResult],
        max_tokens: int | None = None,
        system_instruction: str = "",
        user_query: str = "",
        prompt_overhead_tokens: int = 0,
    ) -> AssembledContext:
        """Assemble retrieved search results into a formatted context block within token limits."""
        budget = max_tokens or self.default_max_tokens

        # 1. Calculate prompt envelope overhead
        prompt_prefix = ""
        if system_instruction:
            prompt_prefix = f"{system_instruction.strip()}\n\n"

        prompt_suffix = ""
        if user_query:
            prompt_suffix = f"\n\n## Developer Query\n{user_query.strip()}\n"

        envelope_tokens = (
            self.count_tokens(prompt_prefix)
            + self.count_tokens(prompt_suffix)
            + prompt_overhead_tokens
        )
        remaining_chunk_budget = max(0, budget - envelope_tokens)

        included: List[SearchResult] = []
        omitted_count = 0
        blocks: List[str] = []
        accumulated_chunk_tokens = 0

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
            chunk_tokens = self.count_tokens(formatted_chunk)

            # Strict budget enforcement: applies to all candidates including the first chunk
            if accumulated_chunk_tokens + chunk_tokens > remaining_chunk_budget:
                omitted_count += 1
                continue

            blocks.append(formatted_chunk)
            accumulated_chunk_tokens += chunk_tokens
            included.append(res)
            seen_spans.add(span_key)

        joined_chunks = "\n".join(blocks).strip()
        context_section = f"## Context & Source References\n\n{joined_chunks}" if joined_chunks else ""
        full_prompt = f"{prompt_prefix}{context_section}{prompt_suffix}".strip()

        estimated_context_tokens = self.count_tokens(joined_chunks)
        total_prompt_tokens = self.count_tokens(full_prompt)

        return AssembledContext(
            formatted_text=joined_chunks,
            full_prompt=full_prompt,
            included_chunks=included,
            omitted_chunks_count=omitted_count,
            estimated_tokens=estimated_context_tokens,
            total_prompt_tokens=total_prompt_tokens,
            max_tokens=budget,
        )
