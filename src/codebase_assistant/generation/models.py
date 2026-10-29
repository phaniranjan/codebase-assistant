"""Data models for code and test proposal generation."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProposalType(str, Enum):
    """Classification of proposed code changes."""

    MODIFICATION = "modification"
    NEW_FILE = "new_file"
    TEST_PROPOSAL = "test_proposal"
    REFACTORING = "refactoring"


class FileProposal(BaseModel):
    """A proposed change to a single target file formatted as a unified diff."""

    target_file: str
    proposal_type: ProposalType
    is_new_file: bool = False
    original_content: Optional[str] = None
    proposed_content: Optional[str] = None
    unified_diff: str = ""
    architectural_reasoning: str = ""
    ast_valid: bool = False
    syntax_errors: List[str] = Field(default_factory=list)
    affected_symbols: List[str] = Field(default_factory=list)


class ProposalResult(BaseModel):
    """Comprehensive proposal generation output containing all affected files and validation status."""

    task_description: str
    provider_name: str
    model_name: str
    proposals: List[FileProposal] = Field(default_factory=list)
    retrieved_chunks_count: int = 0
    total_files_affected: int = 0
    is_valid: bool = False
    verification_summary: str = ""
    token_usage: Optional[Dict[str, int]] = None
