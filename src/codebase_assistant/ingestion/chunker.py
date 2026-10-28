"""Logical chunker transforming parsed symbols into rich CodeChunk models."""

import hashlib
import logging
from typing import List, Optional

from codebase_assistant.ingestion.models import ChunkType, CodeChunk
from codebase_assistant.ingestion.parser import (
    ConfigParser,
    MarkdownParser,
    ParsedSymbol,
    PythonASTParser,
)

logger = logging.getLogger(__name__)

SYMBOL_TO_CHUNK_TYPE = {
    "class": ChunkType.CLASS,
    "function": ChunkType.FUNCTION,
    "method": ChunkType.METHOD,
    "module_header": ChunkType.MODULE_HEADER,
    "doc_section": ChunkType.DOC_SECTION,
    "config_block": ChunkType.CONFIG_BLOCK,
}


class LogicalChunker:
    """Produces logically bounded CodeChunk instances with metadata and line numbers."""

    def __init__(
        self,
        max_chunk_chars: int = 4000,
        chunk_overlap_lines: int = 10,
    ) -> None:
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap_lines = chunk_overlap_lines
        self.python_parser = PythonASTParser()
        self.markdown_parser = MarkdownParser()
        self.config_parser = ConfigParser()

    @staticmethod
    def generate_chunk_id(file_path: str, symbol_name: str, start_line: int) -> str:
        """Create a deterministic unique chunk ID."""
        raw_key = f"{file_path}:{symbol_name}:{start_line}"
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
        safe_sym = symbol_name.replace(".", "_") if symbol_name else "chunk"
        return f"{safe_sym}_{digest}"

    def chunk_file(
        self,
        content: str,
        rel_file_path: str,
        language: str,
        source_revision: Optional[str] = None,
    ) -> List[CodeChunk]:
        """Parse file content and produce logical CodeChunk items."""
        if not content.strip():
            return []

        # 1. Parse into structured symbols
        if language == "python":
            symbols = self.python_parser.parse_source(content, file_path=rel_file_path)
        elif language == "markdown":
            symbols = self.markdown_parser.parse_source(content, file_path=rel_file_path)
        else:
            symbols = self.config_parser.parse_source(content, file_path=rel_file_path)

        # 2. Convert symbols to CodeChunks
        chunks: List[CodeChunk] = []
        for sym in symbols:
            chunk_type = SYMBOL_TO_CHUNK_TYPE.get(sym.symbol_type, ChunkType.UNKNOWN)

            # Check if chunk exceeds max characters and needs sub-chunking
            if len(sym.content) > self.max_chunk_chars:
                sub_chunks = self._subdivide_large_symbol(
                    sym, rel_file_path, language, chunk_type, source_revision
                )
                chunks.extend(sub_chunks)
            else:
                chunk_id = self.generate_chunk_id(rel_file_path, sym.name, sym.start_line)
                content_hash = hashlib.sha256(sym.content.encode("utf-8")).hexdigest()

                chunks.append(
                    CodeChunk(
                        chunk_id=chunk_id,
                        file_path=rel_file_path,
                        content=sym.content,
                        language=language,
                        chunk_type=chunk_type,
                        symbol_name=sym.name,
                        parent_symbol=sym.parent_name,
                        start_line=sym.start_line,
                        end_line=sym.end_line,
                        content_hash=content_hash,
                        source_revision=source_revision,
                        docstring=sym.docstring,
                        dependencies=sym.dependencies,
                        metadata={
                            "decorators": sym.decorators,
                            "parameters": sym.parameters,
                        },
                    )
                )

        return chunks

    def _subdivide_large_symbol(
        self,
        sym: ParsedSymbol,
        file_path: str,
        language: str,
        chunk_type: ChunkType,
        source_revision: Optional[str],
    ) -> List[CodeChunk]:
        """Split oversized symbol into overlapping line windows while preserving metadata."""
        lines = sym.content.splitlines()
        chunks: List[CodeChunk] = []
        step = max(20, (self.max_chunk_chars // 80))
        overlap = self.chunk_overlap_lines

        curr_idx = 0
        part_num = 1

        while curr_idx < len(lines):
            end_idx = min(len(lines), curr_idx + step)
            segment_lines = lines[curr_idx:end_idx]
            segment_content = "\n".join(segment_lines).strip()

            actual_start = sym.start_line + curr_idx
            actual_end = sym.start_line + end_idx - 1
            part_name = f"{sym.name}_part{part_num}"

            chunk_id = self.generate_chunk_id(file_path, part_name, actual_start)
            content_hash = hashlib.sha256(segment_content.encode("utf-8")).hexdigest()

            chunks.append(
                CodeChunk(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    content=segment_content,
                    language=language,
                    chunk_type=chunk_type,
                    symbol_name=part_name,
                    parent_symbol=sym.name,
                    start_line=actual_start,
                    end_line=actual_end,
                    content_hash=content_hash,
                    source_revision=source_revision,
                    docstring=sym.docstring if part_num == 1 else None,
                    dependencies=sym.dependencies,
                    metadata={"sub_part": part_num},
                )
            )

            if end_idx >= len(lines):
                break
            curr_idx += max(1, step - overlap)
            part_num += 1

        return chunks
