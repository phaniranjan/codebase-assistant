"""Unit tests for AST and structural parsers."""

import unittest

from codebase_assistant.ingestion.parser import (
    ConfigParser,
    MarkdownParser,
    PythonASTParser,
)

SAMPLE_PYTHON_CODE = '''"""Module docstring for pipeline testing."""

import math
import os
from typing import List, Optional


class DataProcessor:
    """Processes incoming data batches."""

    def __init__(self, batch_size: int = 32) -> None:
        self.batch_size = batch_size

    def process(self, items: List[str]) -> List[str]:
        """Process items by stripping whitespace."""
        return [item.strip() for item in items]


def standalone_worker(count: int) -> int:
    """Worker function returning double the count."""
    return count * 2
'''


class TestParsers(unittest.TestCase):
    """Test AST symbol extraction and document parsing."""

    def setUp(self):
        self.py_parser = PythonASTParser()
        self.md_parser = MarkdownParser()
        self.cfg_parser = ConfigParser()

    def test_python_ast_parser_classes_and_methods(self):
        symbols = self.py_parser.parse_source(SAMPLE_PYTHON_CODE, file_path="src/processor.py")

        symbol_names = [s.name for s in symbols]
        self.assertIn("processor:module", symbol_names)  # module_header
        self.assertIn("DataProcessor", symbol_names)  # class
        self.assertIn("DataProcessor.__init__", symbol_names)  # method
        self.assertIn("DataProcessor.process", symbol_names)  # method
        self.assertIn("standalone_worker", symbol_names)  # function

        # Verify method details
        process_sym = next(s for s in symbols if s.name == "DataProcessor.process")
        self.assertEqual(process_sym.symbol_type, "method")
        self.assertEqual(process_sym.parent_name, "DataProcessor")
        self.assertEqual(process_sym.docstring, "Process items by stripping whitespace.")
        self.assertTrue(process_sym.start_line > 0)
        self.assertTrue(process_sym.end_line >= process_sym.start_line)

        # Verify module imports
        header_sym = next(s for s in symbols if s.symbol_type == "module_header")
        self.assertIn("math", header_sym.dependencies)
        self.assertIn("os", header_sym.dependencies)
        self.assertIn("typing.List", header_sym.dependencies)

    def test_python_ast_syntax_error_tolerance(self):
        bad_code = "def invalid_syntax( broken:"
        symbols = self.py_parser.parse_source(bad_code, file_path="broken.py")

        self.assertEqual(len(symbols), 1)
        self.assertEqual(symbols[0].symbol_type, "module_header")
        self.assertEqual(symbols[0].content, bad_code)

    def test_markdown_parser_headings(self):
        md_text = """# Architecture Overview
This is the architecture description.

```python
# code example
print("code block with # not treated as heading")
```

## Storage Layer
Details about storage.

### Vector Storage
FAISS details.
"""
        symbols = self.md_parser.parse_source(md_text, file_path="docs/arch.md")

        headings = [s.name for s in symbols]
        self.assertIn("Architecture Overview", headings)
        self.assertIn("Storage Layer", headings)
        self.assertIn("Vector Storage", headings)

        overview_sym = next(s for s in symbols if s.name == "Architecture Overview")
        self.assertIn("print(\"code block with # not treated as heading\")", overview_sym.content)

    def test_config_parser(self):
        cfg_text = """[project]
name = "demo"
version = "1.0.0"
"""
        symbols = self.cfg_parser.parse_source(cfg_text, file_path="pyproject.toml")
        self.assertEqual(len(symbols), 1)
        self.assertEqual(symbols[0].symbol_type, "config_block")
        self.assertIn("name = \"demo\"", symbols[0].content)


if __name__ == "__main__":
    unittest.main()
