"""Deterministic mock Gemini client for fast offline testing and verification."""

import math
import re
from typing import Iterator, Optional

from codebase_assistant.llm.client import BaseLLMClient
from codebase_assistant.llm.models import LLMResponse


class MockGeminiClient(BaseLLMClient):
    """Deterministic, citation-aware mock LLM for offline testing."""

    CHARS_PER_TOKEN = 3.5

    def __init__(
        self,
        model_name: str = "mock-gemini-v1",
        canned_response: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self.canned_response = canned_response

    @property
    def model_name(self) -> str:
        return self._model_name

    def count_tokens(self, text: str) -> int:
        """Estimate token count for mock client."""
        if not text:
            return 0
        return max(1, int(math.ceil(len(text) / self.CHARS_PER_TOKEN)))

    def _synthesize_answer(self, prompt: str) -> str:
        """Synthesize a grounded answer based on the references in the prompt."""
        if self.canned_response is not None:
            return self.canned_response

        # Check if context is missing or empty
        if "No relevant repository chunks were found in the index" in prompt:
            return "The retrieved repository context does not contain sufficient details to answer this question."

        # Extract references in context: ### [Reference 1] `path` (lines X-Y, symbol: `sym`)
        ref_pattern = re.compile(r"### \[Reference \d+\] `([^`]+)` \(lines (\d+)-(\d+)(?:, symbol: `([^`]+)`)?\)")
        matches = ref_pattern.findall(prompt)

        if not matches:
            return "The retrieved repository context does not contain sufficient details to answer this question."

        # Extract developer question
        q_match = re.search(r"## Developer Question:\s*(.+?)(?:\n\nRemember|$)", prompt, re.DOTALL)
        query = q_match.group(1).strip() if q_match else "the query"

        # Synthesize answer citing the found references
        lines = [f"Based on the repository context regarding '{query}':\n"]
        for fpath, s_line, e_line, sym in matches[:3]:
            sym_desc = f" in `{sym}`" if sym else ""
            lines.append(
                f"- Found relevant implementation{sym_desc} at `[{fpath}:{s_line}-{e_line}]`."
            )

        lines.append("\nThe functionality is implemented as shown in the referenced source files above.")
        return "\n".join(lines)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate response synchronously."""
        answer = self._synthesize_answer(prompt)
        return LLMResponse(
            text=answer,
            model_name=self._model_name,
            finish_reason="stop",
        )

    def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> Iterator[str]:
        """Stream response chunk by chunk."""
        answer = self._synthesize_answer(prompt)
        words = answer.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
