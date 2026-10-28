"""AST and structural parsers for Python code, Markdown docs, and configuration files."""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedSymbol:
    """Represents an extracted code symbol (class, method, function, module)."""

    name: str
    symbol_type: str  # "class", "function", "method", "module_header"
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    parent_name: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    content: str = ""


class PythonASTParser:
    """Extracts structural symbols, classes, methods, functions, and imports using Python AST."""

    def __init__(self) -> None:
        pass

    def parse_source(self, code: str, file_path: str = "") -> List[ParsedSymbol]:
        """Parse Python source code and return extracted symbols."""
        symbols: List[ParsedSymbol] = []
        lines = code.splitlines()
        total_lines = len(lines)

        try:
            tree = ast.parse(code, filename=file_path)
        except SyntaxError as e:
            logger.warning("Syntax error parsing %s at line %d: %s. Using fallback.", file_path, e.lineno or 1, e)
            return [
                ParsedSymbol(
                    name=Path(file_path).stem if file_path else "module",
                    symbol_type="module_header",
                    start_line=1,
                    end_line=total_lines,
                    content=code,
                )
            ]

        # 1. Extract imports and module docstring
        module_docstring = ast.get_docstring(tree)
        module_imports: List[str] = []
        first_statement_line = 1

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        module_imports.append(f"{node.module}.{alias.name}")

        # Capture module header chunk (docstring + imports up to first class/function)
        header_end_line = total_lines
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                header_end_line = max(1, node.lineno - 1)
                break

        if header_end_line > 0 and (module_docstring or module_imports or header_end_line < total_lines):
            header_lines = lines[:header_end_line]
            header_content = "\n".join(header_lines).strip()
            if header_content:
                stem = Path(file_path).stem if file_path else "module"
                symbols.append(
                    ParsedSymbol(
                        name=f"{stem}:module",
                        symbol_type="module_header",
                        start_line=1,
                        end_line=header_end_line,
                        docstring=module_docstring,
                        dependencies=module_imports,
                        content=header_content,
                    )
                )

        # 2. Extract classes, methods, and standalone functions
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                symbols.extend(self._parse_class(node, lines, module_imports))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(self._parse_function(node, lines, module_imports))

        return symbols

    def _parse_class(
        self, node: ast.ClassDef, lines: List[str], module_imports: List[str]
    ) -> List[ParsedSymbol]:
        """Extract a class and its member methods."""
        class_symbols: List[ParsedSymbol] = []
        class_name = node.name
        class_doc = ast.get_docstring(node)
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", node.lineno)

        decorators = [ast.unparse(d) for d in node.decorator_list]
        base_classes = [ast.unparse(b) for b in node.bases]

        # Extract class header (definition, docstring, class attributes up to first method)
        method_nodes = [
            n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        if method_nodes:
            class_header_end = method_nodes[0].lineno - 1
        else:
            class_header_end = end_line

        class_header_lines = lines[start_line - 1 : class_header_end]
        class_header_content = "\n".join(class_header_lines).strip()

        class_symbols.append(
            ParsedSymbol(
                name=class_name,
                symbol_type="class",
                start_line=start_line,
                end_line=class_header_end,
                docstring=class_doc,
                decorators=decorators,
                dependencies=base_classes,
                content=class_header_content,
            )
        )

        # Extract methods
        for method in method_nodes:
            method_start = method.lineno
            method_end = getattr(method, "end_lineno", method.lineno)
            method_doc = ast.get_docstring(method)
            method_decorators = [ast.unparse(d) for d in method.decorator_list]
            params = [a.arg for a in method.args.args]
            method_content = "\n".join(lines[method_start - 1 : method_end]).strip()

            class_symbols.append(
                ParsedSymbol(
                    name=f"{class_name}.{method.name}",
                    symbol_type="method",
                    start_line=method_start,
                    end_line=method_end,
                    docstring=method_doc,
                    parent_name=class_name,
                    decorators=method_decorators,
                    parameters=params,
                    dependencies=base_classes,
                    content=method_content,
                )
            )

        return class_symbols

    def _parse_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: List[str],
        module_imports: List[str],
    ) -> ParsedSymbol:
        """Extract a standalone function."""
        start_line = node.lineno
        end_line = getattr(node, "end_lineno", node.lineno)
        func_doc = ast.get_docstring(node)
        decorators = [ast.unparse(d) for d in node.decorator_list]
        params = [a.arg for a in node.args.args]
        func_content = "\n".join(lines[start_line - 1 : end_line]).strip()

        return ParsedSymbol(
            name=node.name,
            symbol_type="function",
            start_line=start_line,
            end_line=end_line,
            docstring=func_doc,
            decorators=decorators,
            parameters=params,
            dependencies=[],
            content=func_content,
        )


class MarkdownParser:
    """Parses Markdown files into logical sections based on headings."""

    def parse_source(self, text: str, file_path: str = "") -> List[ParsedSymbol]:
        """Chunk markdown by headings while keeping code fences intact."""
        lines = text.splitlines()
        symbols: List[ParsedSymbol] = []
        heading_re = re.compile(r"^(#{1,6})\s+(.+)$")

        current_heading = Path(file_path).stem if file_path else "Overview"
        current_start = 1
        current_lines: List[str] = []
        in_code_fence = False

        for idx, line in enumerate(lines, start=1):
            if line.strip().startswith("```") or line.strip().startswith("~~~"):
                in_code_fence = not in_code_fence

            match = heading_re.match(line)
            if match and not in_code_fence:
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        symbols.append(
                            ParsedSymbol(
                                name=current_heading,
                                symbol_type="doc_section",
                                start_line=current_start,
                                end_line=idx - 1,
                                content=content,
                            )
                        )
                current_heading = match.group(2).strip()
                current_start = idx
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                symbols.append(
                    ParsedSymbol(
                        name=current_heading,
                        symbol_type="doc_section",
                        start_line=current_start,
                        end_line=len(lines),
                        content=content,
                    )
                )

        return symbols


class ConfigParser:
    """Parses configuration, manifests, and requirements files."""

    def parse_source(self, text: str, file_path: str = "") -> List[ParsedSymbol]:
        """Represent configuration file as structured logical blocks."""
        lines = text.splitlines()
        stem = Path(file_path).stem if file_path else "config"
        content = text.strip()

        return [
            ParsedSymbol(
                name=stem,
                symbol_type="config_block",
                start_line=1,
                end_line=len(lines),
                content=content,
            )
        ]
