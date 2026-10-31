"""Diagnostic engine for analyzing test execution outputs, failures, and stack traces."""

import re
from typing import List, Optional
from pydantic import BaseModel, Field


class DiagnosticResult(BaseModel):
    """Structured diagnostic analysis of a test execution result."""
    passed: bool
    exit_code: int = 0
    failure_summary: str = ""
    failing_tests: List[str] = Field(default_factory=list)
    error_type: str = "None"
    stack_trace_snippet: str = ""
    file_locations: List[str] = Field(default_factory=list)
    hypothesized_cause: str = ""
    recommended_fix_direction: str = ""

    def to_feedback_prompt(self) -> str:
        """Format diagnostics as structured prompt feedback for the next repair attempt."""
        lines = [
            "### Test Failure Diagnostics",
            f"- **Error Type**: {self.error_type}",
            f"- **Failure Summary**: {self.failure_summary}",
        ]
        if self.failing_tests:
            lines.append(f"- **Failing Tests**: {', '.join(self.failing_tests[:5])}")
        if self.file_locations:
            lines.append(f"- **Traceback Locations**: {', '.join(self.file_locations[:5])}")
        if self.stack_trace_snippet:
            lines.extend([
                "- **Stack Trace Snippet**:",
                "```text",
                self.stack_trace_snippet.strip(),
                "```",
            ])
        if self.hypothesized_cause:
            lines.append(f"- **Hypothesized Cause**: {self.hypothesized_cause}")
        if self.recommended_fix_direction:
            lines.append(f"- **Recommended Fix Direction**: {self.recommended_fix_direction}")

        return "\n".join(lines)


class DiagnosticEngine:
    """Parses pytest and unittest console outputs into deterministic, structured diagnostics."""

    # Common Python error types
    KNOWN_EXCEPTIONS = [
        "AssertionError",
        "ZeroDivisionError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "TypeError",
        "ValueError",
        "FileNotFoundError",
        "ImportError",
        "ModuleNotFoundError",
        "NameError",
        "UnboundLocalError",
        "NotImplementedError",
        "RuntimeError",
        "SyntaxError",
        "IndentationError",
        "TabError",
        "TimeoutError",
    ]

    @classmethod
    def analyze(cls, test_output: str, exit_code: Optional[int] = None) -> DiagnosticResult:
        """Analyze test runner stdout/stderr output and produce a structured DiagnosticResult."""
        if not test_output or not test_output.strip():
            # If no output and exit code 0, passed
            passed = exit_code == 0 if exit_code is not None else False
            return DiagnosticResult(
                passed=passed,
                exit_code=exit_code or 0,
                failure_summary="Empty test output." if not passed else "",
                error_type="None" if passed else "Unknown",
            )

        # Check for pass indicators
        is_timeout = "Test Execution Timed Out" in test_output
        is_passed = (
            not is_timeout
            and (exit_code == 0 or exit_code is None)
            and ("Test Execution Result: PASSED" in test_output or "OK" in test_output.splitlines()[-3:])
            and ("FAILED" not in test_output and "FAIL:" not in test_output and "ERROR:" not in test_output)
        )

        if is_passed:
            return DiagnosticResult(
                passed=True,
                exit_code=0,
                failure_summary="",
                error_type="None",
            )

        computed_exit_code = exit_code if exit_code is not None else 1
        if is_timeout:
            return DiagnosticResult(
                passed=False,
                exit_code=computed_exit_code,
                failure_summary="Test execution timed out.",
                error_type="TimeoutError",
                hypothesized_cause="The test runner exceeded the configured execution timeout, likely due to an infinite loop or hanging I/O.",
                recommended_fix_direction="Check for unbounded loops, blocking socket calls, or deadlock conditions.",
            )

        failing_tests = cls._extract_failing_tests(test_output)
        error_type = cls._extract_error_type(test_output)
        stack_trace = cls._extract_stack_trace(test_output)
        file_locations = cls._extract_file_locations(test_output)
        summary = cls._extract_failure_summary(test_output, error_type, failing_tests)
        cause, direction = cls._hypothesize(error_type, summary, stack_trace)

        return DiagnosticResult(
            passed=False,
            exit_code=computed_exit_code,
            failure_summary=summary,
            failing_tests=failing_tests,
            error_type=error_type,
            stack_trace_snippet=stack_trace,
            file_locations=file_locations,
            hypothesized_cause=cause,
            recommended_fix_direction=direction,
        )

    @classmethod
    def _extract_failing_tests(cls, output: str) -> List[str]:
        failing: List[str] = []

        # Pytest format: FAILED tests/test_foo.py::test_bar - AssertionError...
        for line in output.splitlines():
            line_str = line.strip()
            if line_str.startswith("FAILED "):
                parts = line_str[7:].split(" - ", 1)
                test_id = parts[0].strip()
                if test_id and test_id not in failing:
                    failing.append(test_id)
            # Unittest format: FAIL: test_bar (tests.test_foo.TestFoo.test_bar)
            elif line_str.startswith("FAIL: ") or line_str.startswith("ERROR: "):
                test_id = line_str.split(": ", 1)[1].strip()
                if test_id and test_id not in failing:
                    failing.append(test_id)

        return failing

    @classmethod
    def _extract_error_type(cls, output: str) -> str:
        for exc in cls.KNOWN_EXCEPTIONS:
            # Check for "E   AssertionError:" or "AssertionError:" or "raise ZeroDivisionError"
            pattern = rf"(?:E\s+|Traceback.*?|[\s\n]|^){exc}(?::|\s)"
            if re.search(pattern, output, re.MULTILINE):
                return exc

        # Check for generic exception
        if "Exception:" in output:
            return "Exception"
        if "SyntaxError" in output:
            return "SyntaxError"
        return "TestFailure"

    @classmethod
    def _extract_stack_trace(cls, output: str) -> str:
        lines = output.splitlines()
        traceback_lines: List[str] = []
        in_traceback = False

        for line in lines:
            if "Traceback (most recent call last):" in line:
                in_traceback = True
                traceback_lines = [line]
                continue
            if line.startswith("___ ") and line.endswith(" ___"):  # Pytest failure header
                in_traceback = True
                traceback_lines = [line]
                continue
            if in_traceback:
                traceback_lines.append(line)
                # Terminate traceback on blank line or next test separator
                if line.startswith("===") or (len(traceback_lines) > 20 and not line.startswith(" ")):
                    break

        if traceback_lines:
            return "\n".join(traceback_lines[:20])

        # Pytest E-lines fallback
        e_lines = [l for l in lines if l.startswith("E   ") or l.startswith(">   ")]
        if e_lines:
            return "\n".join(e_lines[:12])

        return ""

    @classmethod
    def _extract_file_locations(cls, output: str) -> List[str]:
        locations: List[str] = []
        # Pattern: File "...", line 123
        matches = re.findall(r'File\s+["\']([^"\']+)["\'],\s+line\s+(\d+)', output)
        for path_str, line_no in matches:
            loc = f"{path_str}:{line_no}"
            if loc not in locations:
                locations.append(loc)

        # Pytest pattern: path/to/file.py:123:
        pyt_matches = re.findall(r'([\w/\.\\]+\.py):(\d+):', output)
        for path_str, line_no in pyt_matches:
            loc = f"{path_str}:{line_no}"
            if loc not in locations:
                locations.append(loc)

        return locations

    @classmethod
    def _extract_failure_summary(cls, output: str, error_type: str, failing_tests: List[str]) -> str:
        # Search for explicit assertion/error message line
        lines = output.splitlines()
        for line in reversed(lines):
            line_str = line.strip()
            if line_str.startswith(f"{error_type}:") or line_str.startswith(f"E   {error_type}:"):
                clean = line_str.replace("E   ", "").strip()
                if failing_tests:
                    return f"{clean} (in {failing_tests[0]})"
                return clean

        for line in lines:
            if "AssertionError" in line or "assert " in line:
                clean = line.replace("E   ", "").strip()
                return clean

        if failing_tests:
            return f"{error_type} in {failing_tests[0]}"

        return f"Test command failed with {error_type}."

    @classmethod
    def _hypothesize(cls, error_type: str, summary: str, stack_trace: str) -> tuple[str, str]:
        if error_type == "ZeroDivisionError":
            return (
                "Division by zero occurred because a denominator evaluated to 0 without guard validation.",
                "Add defensive checks (e.g. `if b == 0: raise ValueError(...)` or return safe fallback) before dividing.",
            )
        if error_type == "AssertionError":
            return (
                f"Computed value did not match expected test assertion: {summary}.",
                "Inspect the function return calculation and edge cases to ensure return matches expected test specification.",
            )
        if error_type in ("KeyError", "IndexError"):
            return (
                f"Missing dictionary key or out-of-range index accessed: {summary}.",
                "Verify collection length or use `.get()` with a default value before indexing.",
            )
        if error_type in ("TypeError", "AttributeError"):
            return (
                f"Unexpected type or None reference encountered: {summary}.",
                "Check for None values, verify method signature parameter types, and validate attribute presence.",
            )
        if error_type in ("ImportError", "ModuleNotFoundError"):
            return (
                f"Required symbol or module could not be imported: {summary}.",
                "Verify import paths, check for circular imports, and ensure dependencies exist in repository manifest.",
            )
        if error_type in ("SyntaxError", "IndentationError"):
            return (
                f"Generated code contains a syntax or indentation flaw: {summary}.",
                "Ensure valid Python syntax with correct colon placements, matched parentheses, and consistent 4-space indentation.",
            )

        return (
            f"Test failed due to {error_type}.",
            "Inspect the failure stack trace and target code logic to resolve discrepancies.",
        )
