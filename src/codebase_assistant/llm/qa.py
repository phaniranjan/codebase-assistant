"""Orchestrates codebase retrieval, grounded prompt construction, and Gemini Q&A reasoning."""

import logging
from typing import Callable, Iterator, List, Literal, Optional, Tuple

from codebase_assistant.config import get_settings
from codebase_assistant.llm.client import BaseLLMClient, GeminiClient
from codebase_assistant.llm.mock_client import MockGeminiClient
from codebase_assistant.llm.models import Citation, QAResponse
from codebase_assistant.llm.prompts import GroundedPromptBuilder
from codebase_assistant.retrieval.context_assembler import ContextAssembler
from codebase_assistant.retrieval.hybrid import HybridRetriever, SearchResult

logger = logging.getLogger(__name__)


class StreamingQAResponse:
    """Wraps a token stream, accumulating the generated answer and validating citations upon completion."""

    def __init__(
        self,
        stream_iter: Iterator[str],
        query: str,
        included_chunks: List[SearchResult],
        total_retrieved: int,
        is_truncated: bool,
        prompt: str,
        model_name: str,
        token_counter: Callable[[str], int],
        on_complete: Optional[Callable[[QAResponse], None]] = None,
    ) -> None:
        self._iter = stream_iter
        self.query = query
        self.included_chunks = included_chunks
        self.total_retrieved = total_retrieved
        self.is_truncated = is_truncated
        self.prompt = prompt
        self.model_name = model_name
        self.token_counter = token_counter
        self.on_complete = on_complete

        self.accumulated_text: str = ""
        self.response: Optional[QAResponse] = None

    def __iter__(self) -> Iterator[str]:
        parts = []
        for token in self._iter:
            parts.append(token)
            yield token

        self.accumulated_text = "".join(parts)
        citations = GroundedPromptBuilder.extract_citations(
            answer=self.accumulated_text,
            retrieved_chunks=self.included_chunks,
        )
        self.response = QAResponse(
            query=self.query,
            answer=self.accumulated_text,
            citations=citations,
            retrieved_chunks_count=self.total_retrieved,
            assembled_chunks_count=len(self.included_chunks),
            omitted_chunks_count=max(0, self.total_retrieved - len(self.included_chunks)),
            model_used=self.model_name,
            is_truncated=self.is_truncated,
            context_tokens=self.token_counter(self.prompt),
        )

        if self.on_complete:
            self.on_complete(self.response)

    def get_response(self) -> Optional[QAResponse]:
        """Return the validated QAResponse after the stream has completed."""
        return self.response


class CodebaseQAEngine:
    """Coordinates retrieval, context assembly, grounded prompting, and LLM reasoning."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: Optional[BaseLLMClient] = None,
        context_assembler: Optional[ContextAssembler] = None,
        default_budget: int = 8192,
    ) -> None:
        self.retriever = retriever
        self.default_budget = default_budget

        if llm_client is not None:
            self.llm_client = llm_client
        else:
            settings = get_settings()
            if settings.gemini_api_key:
                try:
                    self.llm_client = GeminiClient()
                except Exception as e:
                    logger.warning("Could not initialize GeminiClient (%s). Using MockGeminiClient.", e)
                    self.llm_client = MockGeminiClient()
            else:
                self.llm_client = MockGeminiClient()

        if context_assembler is not None:
            self.assembler = context_assembler
        else:
            self.assembler = ContextAssembler(
                default_max_tokens=self.default_budget,
                token_counter=self.llm_client.count_tokens,
            )

    def _prepare_prompt(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["hybrid", "dense_only", "sparse_only"] = "hybrid",
        budget: Optional[int] = None,
    ) -> Tuple[str, str, List[SearchResult], int, bool]:
        """Retrieve chunks, assemble context, and construct grounded prompt envelope."""
        effective_budget = budget or self.default_budget

        # 1. Hybrid retrieval
        results = self.retriever.search(query=query, top_k=top_k, mode=mode)

        # 2. Grounded system instructions
        system_instruction = GroundedPromptBuilder.build_system_instruction()

        # 3. Context assembly with token budgeting
        assembled = self.assembler.assemble(
            results=results,
            max_tokens=effective_budget,
            system_instruction=system_instruction,
            user_query=query,
        )

        # 4. Format prompt envelope
        prompt = GroundedPromptBuilder.build_grounded_prompt(
            query=query,
            formatted_chunks=assembled.formatted_text,
        )

        return (
            prompt,
            system_instruction,
            assembled.included_chunks,
            len(results),
            assembled.is_truncated(),
        )

    def ask(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["hybrid", "dense_only", "sparse_only"] = "hybrid",
        temperature: float = 0.2,
        budget: Optional[int] = None,
    ) -> QAResponse:
        """Answer developer question grounded in retrieved repository source code."""
        prompt, system_instruction, included_chunks, total_retrieved, is_truncated = self._prepare_prompt(
            query=query, top_k=top_k, mode=mode, budget=budget
        )

        response = self.llm_client.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )

        citations = GroundedPromptBuilder.extract_citations(
            answer=response.text,
            retrieved_chunks=included_chunks,
        )

        return QAResponse(
            query=query,
            answer=response.text,
            citations=citations,
            retrieved_chunks_count=total_retrieved,
            assembled_chunks_count=len(included_chunks),
            omitted_chunks_count=max(0, total_retrieved - len(included_chunks)),
            model_used=self.llm_client.model_name,
            is_truncated=is_truncated,
            context_tokens=self.llm_client.count_tokens(prompt),
        )

    def ask_stream(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["hybrid", "dense_only", "sparse_only"] = "hybrid",
        temperature: float = 0.2,
        budget: Optional[int] = None,
        on_complete: Optional[Callable[[QAResponse], None]] = None,
    ) -> StreamingQAResponse:
        """Stream answer tokens for developer question with post-stream citation validation."""
        prompt, system_instruction, included_chunks, total_retrieved, is_truncated = self._prepare_prompt(
            query=query, top_k=top_k, mode=mode, budget=budget
        )

        raw_stream = self.llm_client.generate_stream(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )

        return StreamingQAResponse(
            stream_iter=raw_stream,
            query=query,
            included_chunks=included_chunks,
            total_retrieved=total_retrieved,
            is_truncated=is_truncated,
            prompt=prompt,
            model_name=self.llm_client.model_name,
            token_counter=self.llm_client.count_tokens,
            on_complete=on_complete,
        )
