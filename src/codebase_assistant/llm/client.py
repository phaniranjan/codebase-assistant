"""Gemini LLM client integration using official google-genai SDK."""

from abc import ABC, abstractmethod
import logging
import math
from typing import Iterator, Optional

from codebase_assistant.config import get_settings
from codebase_assistant.llm.models import LLMResponse

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract interface for LLM generation clients."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the active model name."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        """Execute unary content generation."""
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> Iterator[str]:
        """Stream generated text chunks iteratively."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return token count for a text string."""
        pass


class GeminiClient(BaseLLMClient):
    """Production Gemini client using the official google-genai SDK."""

    CHARS_PER_TOKEN = 3.5

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key or settings.gemini_api_key
        self._model_name = model_name or settings.gemini_model or "gemini-2.5-flash"
        self._client = None

    def _get_client(self):
        """Lazily initialize google-genai Client."""
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "GEMINI_API_KEY is required to generate responses. "
                    "Set GEMINI_API_KEY environment variable or pass api_key to the client, "
                    "or use '--provider mock' for offline execution."
                )
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
            except ImportError as e:
                raise ImportError(
                    "google-genai package is required. Install it with: pip install google-genai"
                ) from e
        return self._client

    @property
    def model_name(self) -> str:
        return self._model_name

    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini API when available, falling back to conservative character heuristic."""
        if not text:
            return 0
        if self._api_key:
            try:
                client = self._get_client()
                resp = client.models.count_tokens(model=self._model_name, contents=text)
                if hasattr(resp, "total_tokens") and resp.total_tokens is not None:
                    return resp.total_tokens
            except Exception:
                pass
        return max(1, int(math.ceil(len(text) / self.CHARS_PER_TOKEN)))

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> LLMResponse:
        """Execute synchronous completion with Gemini."""
        client = self._get_client()
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
        )

        try:
            response = client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=config,
            )
            text = response.text or ""
            return LLMResponse(
                text=text,
                model_name=self._model_name,
                finish_reason="stop",
            )
        except Exception as e:
            logger.error("Gemini generate_content failed: %s", e)
            raise RuntimeError(f"Gemini API error: {e}") from e

    def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
    ) -> Iterator[str]:
        """Stream chunks from Gemini generate_content_stream."""
        client = self._get_client()
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
        )

        try:
            response_stream = client.models.generate_content_stream(
                model=self._model_name,
                contents=prompt,
                config=config,
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error("Gemini generate_content_stream failed: %s", e)
            raise RuntimeError(f"Gemini streaming API error: {e}") from e
