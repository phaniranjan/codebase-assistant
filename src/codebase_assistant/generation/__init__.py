"""Code and test proposal generation module."""

from codebase_assistant.generation.code_generator import CodeProposalGenerator
from codebase_assistant.generation.diff_engine import DiffEngine
from codebase_assistant.generation.models import (
    FileProposal,
    ProposalResult,
    ProposalType,
)
from codebase_assistant.generation.test_generator import TestProposalGenerator

__all__ = [
    "DiffEngine",
    "CodeProposalGenerator",
    "TestProposalGenerator",
    "FileProposal",
    "ProposalResult",
    "ProposalType",
]
