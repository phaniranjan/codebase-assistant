"""Schema and validation utilities for the golden evaluation dataset."""

from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class EvalCategory(str, Enum):
    """Evaluation test case categories."""

    ARCHITECTURE = "architecture"
    SYMBOL_LOOKUP = "symbol_lookup"
    DATA_FLOW = "data_flow"
    CONFIGURATION = "configuration"
    ERROR_HANDLING = "error_handling"
    UNANSWERABLE = "unanswerable"
    PROMPT_INJECTION = "prompt_injection"


def normalize_repo_path(path_str: str) -> str:
    """Normalize file paths across Windows and Unix platforms."""
    cleaned = path_str.replace("\\", "/").strip()
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


class GoldenTestCase(BaseModel):
    """A single representative evaluation case grounded in target codebase facts."""

    id: str
    question: str
    category: EvalCategory
    is_answerable: bool = True
    expected_files: List[str] = Field(default_factory=list)
    expected_symbols: List[str] = Field(default_factory=list)
    expected_facts: List[str] = Field(default_factory=list)
    contains_prompt_injection: bool = False
    dataset_version: str = "1.0.0"

    @field_validator("expected_files", mode="before")
    @classmethod
    def _normalize_files(cls, files: Any) -> List[str]:
        if not files:
            return []
        return [normalize_repo_path(str(f)) for f in files]


class GoldenDataset(BaseModel):
    """Versioned collection of golden test cases for codebase evaluation."""

    version: str = "1.0.0"
    target_repository: str = "ai-evaluation"
    description: str = "Golden evaluation dataset for retrieval and grounded Q&A benchmarking."
    cases: List[GoldenTestCase] = Field(default_factory=list)

    @classmethod
    def load_from_file(cls, file_path: Path | str) -> "GoldenDataset":
        """Load and parse golden dataset from JSON file."""
        p = Path(file_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Golden dataset file not found: {p}")

        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def save_to_file(self, file_path: Path | str) -> None:
        """Persist dataset to JSON with deterministic indentation."""
        p = Path(file_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def filter_cases(
        self,
        category: Optional[EvalCategory | str] = None,
        is_answerable: Optional[bool] = None,
        contains_prompt_injection: Optional[bool] = None,
    ) -> List[GoldenTestCase]:
        """Filter cases matching specified attributes."""
        matched = self.cases
        if category is not None:
            cat_val = category.value if isinstance(category, EvalCategory) else category
            matched = [c for c in matched if c.category.value == cat_val]
        if is_answerable is not None:
            matched = [c for c in matched if c.is_answerable == is_answerable]
        if contains_prompt_injection is not None:
            matched = [c for c in matched if c.contains_prompt_injection == contains_prompt_injection]
        return matched

    def validate_against_repo(self, repo_path: Path | str) -> List[str]:
        """Verify that expected files in answerable cases actually exist in target repository.

        Returns:
            List of error messages. Empty list indicates 100% valid grounding.
        """
        r_path = Path(repo_path).resolve()
        errors: List[str] = []

        for case in self.cases:
            if not case.is_answerable:
                continue

            if not case.expected_files:
                errors.append(f"Case '{case.id}' is answerable but defines zero expected_files.")

            for rel_file in case.expected_files:
                target_f = r_path / rel_file
                if not target_f.is_file():
                    errors.append(f"Case '{case.id}' references non-existent file: '{rel_file}'")

        return errors
