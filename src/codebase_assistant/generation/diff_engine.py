"""Deterministic diff engine, in-memory patch applicator, and AST validator."""

import ast
import difflib
from pathlib import Path
import re
from typing import List, Optional, Tuple


def normalize_rel_path(path_str: str) -> str:
    """Normalize file paths across Windows and Unix platforms."""
    cleaned = path_str.replace("\\", "/").strip()
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.lstrip("/")


class DiffEngine:
    """Provides unified diff generation, in-memory dry-run application, and syntax validation."""

    @classmethod
    def generate_unified_diff(
        cls,
        original_content: str,
        proposed_content: str,
        file_path: str,
        is_new_file: bool = False,
    ) -> str:
        """Produce a standard git-compatible unified diff string."""
        norm_path = normalize_rel_path(file_path)
        from_file = "/dev/null" if is_new_file else f"a/{norm_path}"
        to_file = f"b/{norm_path}"

        orig_lines = original_content.splitlines(keepends=True)
        prop_lines = proposed_content.splitlines(keepends=True)

        diff_lines = list(
            difflib.unified_diff(
                orig_lines,
                prop_lines,
                fromfile=from_file,
                tofile=to_file,
                lineterm="\n",
            )
        )
        return "".join(diff_lines)

    @classmethod
    def validate_target_path(cls, repository_path: Path | str, target_file: str) -> Path:
        """Enforce workspace containment and prevent path traversal attacks."""
        repo = Path(repository_path).resolve()
        norm = normalize_rel_path(target_file)

        # Disallow parent references
        if ".." in Path(norm).parts:
            raise ValueError(f"Path traversal detected: '{target_file}' attempts to navigate outside repository.")

        target = (repo / norm).resolve()
        try:
            target.relative_to(repo)
        except ValueError:
            raise ValueError(f"Path traversal detected: '{target}' resolves outside repository boundary '{repo}'.")

        return target

    @classmethod
    def validate_ast_syntax(cls, code: str, filename: str = "<proposal>") -> Tuple[bool, List[str]]:
        """Deterministically verify whether Python code parses into a valid AST."""
        if not code.strip():
            return True, []
        try:
            ast.parse(code, filename=filename)
            return True, []
        except SyntaxError as e:
            msg = f"SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}"
            if e.text:
                msg += f" -> '{e.text.strip()}'"
            return False, [msg]
        except Exception as e:
            return False, [f"AST parse error: {str(e)}"]

    @classmethod
    def apply_diff_in_memory(
        cls,
        original_content: str,
        unified_diff: str,
    ) -> str:
        """Deterministically apply a unified diff to original text in memory without touching disk.

        Raises ValueError if hunks fail to match context or are malformed.
        """
        if not unified_diff.strip():
            return original_content

        diff_lines = unified_diff.splitlines()
        orig_lines = original_content.splitlines()

        # If it's a creation of a new file from /dev/null
        is_new_file = any(line.startswith("--- /dev/null") for line in diff_lines[:5])
        if is_new_file and not original_content.strip():
            added_lines = [
                line[1:] for line in diff_lines
                if line.startswith("+") and not line.startswith("+++")
            ]
            return "\n".join(added_lines) + ("\n" if added_lines else "")

        hunk_header_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        hunks = []
        current_hunk = None

        for line in diff_lines:
            match = hunk_header_re.match(line)
            if match:
                if current_hunk:
                    hunks.append(current_hunk)
                old_start = int(match.group(1))
                old_count = int(match.group(2)) if match.group(2) is not None else 1
                new_start = int(match.group(3))
                new_count = int(match.group(4)) if match.group(4) is not None else 1
                current_hunk = {
                    "old_start": old_start,
                    "old_count": old_count,
                    "new_start": new_start,
                    "new_count": new_count,
                    "lines": [],
                }
            elif current_hunk is not None:
                if line.startswith(("-", "+", " ", "\\")):
                    current_hunk["lines"].append(line)

        if current_hunk:
            hunks.append(current_hunk)

        if not hunks:
            return original_content

        result_lines = []
        orig_idx = 0

        for hunk in hunks:
            # 1-indexed to 0-indexed
            target_old_start = max(0, hunk["old_start"] - 1)

            # Copy unchanged lines leading up to this hunk
            while orig_idx < target_old_start and orig_idx < len(orig_lines):
                result_lines.append(orig_lines[orig_idx])
                orig_idx += 1

            for h_line in hunk["lines"]:
                prefix = h_line[0]
                content = h_line[1:]

                if prefix == " ":
                    if orig_idx < len(orig_lines):
                        if orig_lines[orig_idx] != content:
                            raise ValueError(
                                f"Patch context mismatch at line {orig_idx + 1}: "
                                f"expected '{content}', found '{orig_lines[orig_idx]}'"
                            )
                        result_lines.append(orig_lines[orig_idx])
                        orig_idx += 1
                    else:
                        result_lines.append(content)
                elif prefix == "-":
                    if orig_idx < len(orig_lines):
                        if orig_lines[orig_idx] != content:
                            raise ValueError(
                                f"Patch deletion mismatch at line {orig_idx + 1}: "
                                f"expected to delete '{content}', found '{orig_lines[orig_idx]}'"
                            )
                        orig_idx += 1
                    else:
                        raise ValueError(f"Patch deletion failed: line {orig_idx + 1} exceeds file length")
                elif prefix == "+":
                    result_lines.append(content)
                elif prefix == "\\":
                    # Ignore '\ No newline at end of file'
                    pass

        # Copy any remaining original lines
        while orig_idx < len(orig_lines):
            result_lines.append(orig_lines[orig_idx])
            orig_idx += 1

        output = "\n".join(result_lines)
        if original_content.endswith("\n") or not original_content:
            output += "\n"
        return output
