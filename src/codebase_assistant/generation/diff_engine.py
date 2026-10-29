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
        """Enforce workspace containment and prevent path traversal and symlink escape attacks."""
        repo = Path(repository_path).resolve()

        # Disallow raw parent references in unnormalized input
        raw_parts = Path(target_file).parts
        if ".." in raw_parts:
            raise ValueError(f"Path traversal detected: '{target_file}' attempts to navigate outside repository.")

        # Check for absolute or root-anchored paths (e.g. /path or \path or C:\path)
        target_p = Path(target_file)
        is_root_anchored = target_p.is_absolute() or target_file.startswith(("/", "\\"))

        if is_root_anchored:
            resolved_abs = target_p.resolve()
            try:
                resolved_abs.relative_to(repo)
                target = resolved_abs
            except ValueError:
                raise ValueError(f"Path security violation: Absolute path '{target_file}' resolves outside repository boundary '{repo}'.")
        else:
            norm = normalize_rel_path(target_file)
            if ".." in Path(norm).parts:
                raise ValueError(f"Path traversal detected: '{target_file}' attempts to navigate outside repository.")
            target = (repo / norm).resolve()

        try:
            target.relative_to(repo)
        except ValueError:
            raise ValueError(f"Path traversal or symlink escape detected: '{target}' resolves outside repository boundary '{repo}'.")

        return target

    @classmethod
    def validate_ast_syntax(
        cls,
        code: str,
        filename: str = "<proposal>",
        allow_empty: bool = False,
    ) -> Tuple[bool, List[str]]:
        """Deterministically verify whether Python code parses into a valid AST.

        By default, empty or whitespace-only code is rejected as an invalid proposal.
        """
        if not code or not code.strip():
            if allow_empty:
                return True, []
            return False, ["Code proposal is empty or contains only whitespace."]
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

        Raises ValueError if hunks fail to match context, have mismatched line counts, or are malformed.
        """
        if not unified_diff.strip():
            return original_content

        diff_lines = unified_diff.splitlines()
        orig_lines = original_content.splitlines()

        # Check if this patch creates a new file from /dev/null
        is_new_file = any(line.startswith("--- /dev/null") for line in diff_lines[:5])
        if is_new_file and original_content.strip():
            raise ValueError("Cannot apply new-file patch (--- /dev/null) to non-empty original content")

        hunk_header_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
        hunks = []
        current_hunk = None

        for line_num, line in enumerate(diff_lines, start=1):
            if line.startswith("@@"):
                match = hunk_header_re.match(line)
                if not match:
                    raise ValueError(f"Malformed hunk header syntax at line {line_num}: '{line}'")
                if current_hunk:
                    hunks.append(current_hunk)
                old_start = int(match.group(1))
                old_count = int(match.group(2)) if match.group(2) is not None else 1
                new_start = int(match.group(3))
                new_count = int(match.group(4)) if match.group(4) is not None else 1
                current_hunk = {
                    "header_line": line_num,
                    "old_start": old_start,
                    "old_count": old_count,
                    "new_start": new_start,
                    "new_count": new_count,
                    "lines": [],
                }
            elif current_hunk is not None:
                if line.startswith(("-", "+", " ", "\\")):
                    current_hunk["lines"].append(line)
                elif line.strip():
                    raise ValueError(f"Malformed patch line at diff line {line_num}: '{line}'")

        if current_hunk:
            hunks.append(current_hunk)

        if is_new_file and not hunks:
            raise ValueError("Malformed new-file patch: missing hunk header (@@ ... @@)")

        if not hunks:
            return original_content

        # Hunk validation: verify line counts and structure
        for hunk in hunks:
            if is_new_file:
                if hunk["old_count"] != 0:
                    raise ValueError(
                        f"Malformed new-file patch at diff line {hunk['header_line']}: "
                        f"header specifies old_count={hunk['old_count']} but source is /dev/null"
                    )
                if any(l.startswith(("-", " ")) for l in hunk["lines"]):
                    raise ValueError(
                        f"Malformed new-file patch at diff line {hunk['header_line']}: "
                        f"cannot delete or match context lines for /dev/null source"
                    )

            actual_old = sum(1 for l in hunk["lines"] if l.startswith((" ", "-")))
            actual_new = sum(1 for l in hunk["lines"] if l.startswith((" ", "+")))
            if actual_old != hunk["old_count"]:
                raise ValueError(
                    f"Malformed hunk at diff line {hunk['header_line']}: header specifies {hunk['old_count']} old lines, "
                    f"but hunk contains {actual_old} (@@ -{hunk['old_start']},{hunk['old_count']} +{hunk['new_start']},{hunk['new_count']} @@)"
                )
            if actual_new != hunk["new_count"]:
                raise ValueError(
                    f"Malformed hunk at diff line {hunk['header_line']}: header specifies {hunk['new_count']} new lines, "
                    f"but hunk contains {actual_new} (@@ -{hunk['old_start']},{hunk['old_count']} +{hunk['new_start']},{hunk['new_count']} @@)"
                )

        result_lines = []
        orig_idx = 0

        for hunk in hunks:
            if hunk["old_count"] == 0:
                # For pure insertion (old_count == 0), hunk['old_start'] is the line after which
                # lines are inserted (or 0 if inserted before line 1)
                target_old_start = hunk["old_start"]
                if target_old_start > len(orig_lines):
                    raise ValueError(
                        f"Patch hunk insertion position {hunk['old_start']} exceeds file length {len(orig_lines)}"
                    )
            else:
                target_old_start = max(0, hunk["old_start"] - 1)
                if target_old_start > len(orig_lines):
                    raise ValueError(
                        f"Patch hunk start line {hunk['old_start']} exceeds file length {len(orig_lines)}"
                    )

            # Check for overlapping or out-of-order hunks
            if orig_idx > target_old_start:
                raise ValueError(
                    f"Malformed patch: overlapping or out-of-order hunk at line {hunk['old_start']} "
                    f"(current position is line {orig_idx + 1})"
                )

            # Copy unchanged lines leading up to this hunk
            while orig_idx < target_old_start and orig_idx < len(orig_lines):
                result_lines.append(orig_lines[orig_idx])
                orig_idx += 1

            for h_line in hunk["lines"]:
                prefix = h_line[0]
                content = h_line[1:]

                if prefix == " ":
                    if orig_idx >= len(orig_lines):
                        raise ValueError(
                            f"Patch context mismatch: context line '{content}' occurs past end of original content (line {orig_idx + 1})"
                        )
                    if orig_lines[orig_idx] != content:
                        raise ValueError(
                            f"Patch context mismatch at line {orig_idx + 1}: "
                            f"expected '{content}', found '{orig_lines[orig_idx]}'"
                        )
                    result_lines.append(orig_lines[orig_idx])
                    orig_idx += 1
                elif prefix == "-":
                    if orig_idx >= len(orig_lines):
                        raise ValueError(
                            f"Patch deletion failed: line {orig_idx + 1} exceeds file length"
                        )
                    if orig_lines[orig_idx] != content:
                        raise ValueError(
                            f"Patch deletion mismatch at line {orig_idx + 1}: "
                            f"expected to delete '{content}', found '{orig_lines[orig_idx]}'"
                        )
                    orig_idx += 1
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
        if not output and not original_content:
            return ""
        if original_content.endswith("\n") or not original_content:
            output += "\n"
        return output
