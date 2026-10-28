"""Configuration management for Codebase Assistant."""

import os
from pathlib import Path
from typing import List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application settings with environment variable support."""

    # Gemini API Configuration
    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    gemini_embedding_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_API_KEY") or os.getenv("GEMINI_API_KEY")
    )

    # Models
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    gemini_embedding_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    )

    # Local Cache & Index Storage
    cache_dir: str = Field(
        default_factory=lambda: os.getenv("CODEBASE_CACHE_DIR", ".codebase_assistant")
    )

    # Context & Retrieval Limits
    max_context_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_TOKENS", "8192"))
    )
    default_top_k: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_TOP_K", "5"))
    )

    # Agent Limits & Stopping Conditions
    max_tool_calls_per_turn: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "5"))
    )
    max_reasoning_steps: int = Field(
        default_factory=lambda: int(os.getenv("MAX_REASONING_STEPS", "10"))
    )
    command_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("COMMAND_TIMEOUT_SECONDS", "30"))
    )
    require_confirmation_for_mutations: bool = Field(
        default_factory=lambda: os.getenv("REQUIRE_CONFIRMATION_FOR_MUTATIONS", "true").lower() in ("true", "1", "yes")
    )

    # Default file exclusion patterns
    default_ignore_patterns: List[str] = [
        ".git",
        ".git/**",
        "__pycache__",
        "__pycache__/**",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache",
        ".pytest_cache/**",
        ".mypy_cache",
        ".mypy_cache/**",
        ".venv",
        ".venv/**",
        "venv",
        "venv/**",
        "deepeval_venv",
        "deepeval_venv/**",
        "build",
        "build/**",
        "dist",
        "dist/**",
        "*.egg-info",
        "*.egg-info/**",
        ".codebase_assistant",
        ".codebase_assistant/**",
        "data/indexes/**",
        "*.log",
    ]

    # Secret and credential file patterns (always excluded)
    secret_patterns: List[str] = [
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "*.pkcs12",
        "*.pfx",
        "id_rsa",
        "id_rsa.*",
        "id_ed25519",
        "id_ed25519.*",
        "credentials.json",
        "client_secret*.json",
        "token.json",
    ]


def get_settings() -> Settings:
    """Retrieve global settings instance."""
    return Settings()
