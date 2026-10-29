"""Deterministic mock Gemini client for fast offline testing and verification."""

import math
from pathlib import Path
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

    STOPWORDS = {
        "what", "is", "the", "how", "are", "in", "and", "where", "does", "do",
        "which", "why", "who", "when", "with", "from", "for", "that", "this",
        "about", "into", "over", "after", "before", "between", "under", "system",
        "instruction", "override", "policy", "please", "tell", "explain", "describe",
        "repository", "codebase", "code", "file", "files", "project", "directory",
        "implementation", "implemented", "configure", "configured", "configuration",
        "defined", "define", "works", "work", "using", "used", "handling", "handle",
        "api", "secret", "secrets", "loop", "loops", "network", "networks", "flow",
    }

    def _sanitize_query(self, query: str) -> str:
        """Strip adversarial injection payloads from user query."""
        cleaned = re.sub(r"#[^\n.]*[\n.]", " ", query)
        cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def _synthesize_answer(self, prompt: str) -> str:
        """Synthesize a grounded answer or code proposal based on prompt type."""
        if self.canned_response is not None:
            return self.canned_response

        # 1. Handle Code Proposal Generation
        if "[DEVELOPER TASK]" in prompt:
            task_match = re.search(r"\[DEVELOPER TASK\]\s*(.+?)\s*\[END DEVELOPER TASK\]", prompt, re.DOTALL)
            task_str = task_match.group(1).strip() if task_match else "propose change"
            file_match = re.search(r"Target file to modify or create: '([^']+)'", prompt)
            target_file = file_match.group(1) if file_match else "src/proposal.py"

            # Check context block for existing code
            ctx_match = re.search(r"\[CONTEXT_START\](.*?)\[CONTEXT_END\]", prompt, re.DOTALL)
            ctx_text = ctx_match.group(1) if ctx_match else ""
            code_blocks = re.findall(r"```python\s*(.+?)```", ctx_text, re.DOTALL)
            ctx_code = code_blocks[0].strip() if code_blocks else ""

            task_clean = task_str.split("\n")[0].strip()

            if ctx_code:
                proposed_code = (
                    f"{ctx_code}\n\n"
                    "def proposed_enhancement():\n"
                    f"    \"\"\"Implementation for: {task_clean}\"\"\"\n"
                    "    return True\n"
                )
            else:
                proposed_code = (
                    "\"\"\"Proposed module implementation.\"\"\"\n\n"
                    "def execute_task():\n"
                    f"    \"\"\"Implementation for: {task_clean}\"\"\"\n"
                    "    return {'status': 'success'}\n"
                )

            return (
                "--- ARCHITECTURAL REASONING ---\n"
                f"The proposed modification implements '{task_str}' in '{target_file}' with structured error handling and AST-valid syntax.\n\n"
                f"--- FILE: {target_file} ---\n"
                "```python\n"
                f"{proposed_code}\n"
                "```"
            )

        # 2. Handle Test Proposal Generation
        if "[TEST GENERATION TASK]" in prompt:
            file_match = re.search(r"Target source file: '([^']+)'", prompt)
            target_file = file_match.group(1) if file_match else "src/module.py"
            mod_name = Path(target_file).stem
            cls_name = "".join(part.capitalize() for part in mod_name.split("_"))

            test_code = (
                f"\"\"\"Targeted unit tests for {target_file}.\"\"\"\n\n"
                "import unittest\n\n\n"
                f"class Test{cls_name}(unittest.TestCase):\n"
                f"    \"\"\"Test suite covering core methods and edge cases of {mod_name}.\"\"\"\n\n"
                "    def test_core_functionality(self):\n"
                "        self.assertTrue(True)\n\n"
                "    def test_edge_case_empty_input(self):\n"
                "        self.assertIsNotNone(True)\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            )

            return (
                "--- ARCHITECTURAL REASONING ---\n"
                f"Generated targeted unit tests covering core execution and boundary edge cases for '{target_file}'.\n\n"
                f"--- FILE: tests/unit/test_{mod_name}.py ---\n"
                "```python\n"
                f"{test_code}\n"
                "```"
            )

        # 3. Standard Grounded Q&A Synthesizer

        # Check if context is missing or empty
        if "No relevant repository chunks were found in the index" in prompt:
            return "The retrieved repository context does not contain sufficient details to answer this question."

        # Extract untrusted context block
        ctx_match = re.search(r"<UNTRUSTED_REPOSITORY_CONTEXT>(.*?)</UNTRUSTED_REPOSITORY_CONTEXT>", prompt, re.DOTALL)
        context_text = ctx_match.group(1).lower() if ctx_match else ""

        # Extract developer question
        q_match = re.search(r"## Developer Question:\s*(.+?)(?:\n\nRemember|$)", prompt, re.DOTALL)
        raw_query = q_match.group(1).strip() if q_match else "the query"
        clean_query = self._sanitize_query(raw_query)

        # Extract query keywords
        query_words = re.findall(r"\b[a-zA-Z0-9_]{3,}\b", clean_query.lower())
        sig_words = [w for w in query_words if w not in self.STOPWORDS]

        # Check if context contains evidence for any significant query word
        has_evidence = any(w in context_text for w in sig_words) if sig_words else False

        # Extract references in context: ### [Reference 1] `path` (lines X-Y, symbol: `sym`)
        ref_pattern = re.compile(r"### \[Reference \d+\] `([^`]+)` \(lines (\d+)-(\d+)(?:, symbol: `([^`]+)`)?\)")
        matches = ref_pattern.findall(prompt)

        # If no references or no evidence found for query keywords, abstain!
        if not matches or not has_evidence:
            return "The retrieved repository context does not contain sufficient details to answer this question."

        # Filter references that actually contain significant query terms
        relevant_matches = []
        for fpath, s_line, e_line, sym in matches:
            sym_lower = sym.lower() if sym else ""
            fpath_lower = fpath.lower()
            if any(w in fpath_lower or w in sym_lower for w in sig_words):
                relevant_matches.append((fpath, s_line, e_line, sym))

        if not relevant_matches:
            relevant_matches = matches[:2]

        # Synthesize answer citing verified references without echoing adversarial payloads
        lines = [f"Based on the repository context regarding '{clean_query}':\n"]
        for fpath, s_line, e_line, sym in relevant_matches[:3]:
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
