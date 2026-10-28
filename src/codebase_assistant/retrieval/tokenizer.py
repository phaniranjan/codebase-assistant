"""Code-aware tokenizer supporting camelCase, snake_case, dot notation, and punctuation."""

import re
from typing import List, Set

# Common programming stop words that carry little retrieval value
CODE_STOP_WORDS: Set[str] = {
    "self",
    "cls",
    "this",
    "none",
    "true",
    "false",
    "return",
    "def",
    "class",
    "import",
    "from",
    "as",
    "pass",
}


class CodeTokenizer:
    """Extracts search tokens from code, identifiers, and natural language queries."""

    # Regex for splitting camelCase and PascalCase
    CAMEL_REGEX = re.compile(r"([a-z0-9])([A-Z])|([A-Z]+)([A-Z][a-z0-9])")
    # Regex for identifier matching
    IDENT_REGEX = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

    @classmethod
    def split_identifier(cls, identifier: str) -> List[str]:
        """Split a compound identifier (camelCase, snake_case, or mixed) into constituent parts."""
        # Split on snake_case underscores first
        parts = identifier.split("_")
        sub_tokens: List[str] = []

        for p in parts:
            if not p:
                continue
            # Split camelCase
            expanded = cls.CAMEL_REGEX.sub(r"\1\3 \2\4", p)
            for token in expanded.split():
                clean = token.lower()
                if len(clean) > 1 and clean not in CODE_STOP_WORDS:
                    sub_tokens.append(clean)

        return sub_tokens

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenize code or query string into a list of normalized tokens."""
        if not text:
            return []

        tokens: List[str] = []
        raw_words = cls.IDENT_REGEX.findall(text)

        for word in raw_words:
            lowered = word.lower()
            if len(lowered) > 1 and lowered not in CODE_STOP_WORDS:
                tokens.append(lowered)

            # Sub-identifiers (e.g. 'RepositoryScanner' -> ['repository', 'scanner'])
            sub_parts = cls.split_identifier(word)
            for sub in sub_parts:
                if sub != lowered:
                    tokens.append(sub)

        return tokens
