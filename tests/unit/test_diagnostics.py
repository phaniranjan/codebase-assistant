"""Unit tests for DiagnosticEngine test output parsing and diagnostic synthesis."""

import unittest

from codebase_assistant.agent.diagnostics import DiagnosticEngine, DiagnosticResult


class TestDiagnosticEngine(unittest.TestCase):
    """Test suite verifying pytest and unittest failure extraction and diagnostics."""

    def test_passed_output(self):
        output = (
            "Test Execution Result: PASSED\n"
            "--- STDOUT ---\n"
            "test_add (tests.unit.test_calc.TestCalc.test_add) ... ok\n\n"
            "----------------------------------------------------------------------\n"
            "Ran 1 test in 0.001s\n\n"
            "OK\n"
            "--- STDERR ---\n(empty)"
        )
        diag = DiagnosticEngine.analyze(output, exit_code=0)
        self.assertTrue(diag.passed)
        self.assertEqual(diag.exit_code, 0)
        self.assertEqual(diag.error_type, "None")
        self.assertEqual(diag.failure_summary, "")

    def test_pytest_assertion_failure(self):
        output = (
            "Test Execution Result: FAILED (Exit Code 1)\n"
            "--- STDOUT ---\n"
            "=== FAILURES ===\n"
            "___ TestCalc.test_divide ___\n"
            "tests/unit/test_calc.py:15: in test_divide\n"
            "    assert divide(10, 2) == 4\n"
            "E   AssertionError: assert 5 == 4\n"
            "FAILED tests/unit/test_calc.py::TestCalc::test_divide - AssertionError: assert 5 == 4\n"
            "=== 1 failed in 0.05s ==="
        )
        diag = DiagnosticEngine.analyze(output, exit_code=1)
        self.assertFalse(diag.passed)
        self.assertEqual(diag.exit_code, 1)
        self.assertEqual(diag.error_type, "AssertionError")
        self.assertIn("AssertionError", diag.failure_summary)
        self.assertIn("tests/unit/test_calc.py::TestCalc::test_divide", diag.failing_tests)
        self.assertIn("AssertionError", diag.hypothesized_cause)
        self.assertTrue(len(diag.file_locations) > 0)

        # Check prompt formatting
        feedback = diag.to_feedback_prompt()
        self.assertIn("Test Failure Diagnostics", feedback)
        self.assertIn("AssertionError", feedback)

    def test_unittest_zero_division_failure(self):
        output = (
            "Test Execution Result: FAILED (Exit Code 1)\n"
            "--- STDOUT ---\n"
            "FAIL: test_div_zero (tests.unit.test_calc.TestCalc.test_div_zero)\n"
            "Traceback (most recent call last):\n"
            '  File "tests/unit/test_calc.py", line 22, in test_div_zero\n'
            "    div(10, 0)\n"
            '  File "src/calc.py", line 12, in div\n'
            "    return a / b\n"
            "ZeroDivisionError: division by zero\n\n"
            "----------------------------------------------------------------------\n"
            "Ran 1 test in 0.002s\n\n"
            "FAILED (failures=1)"
        )
        diag = DiagnosticEngine.analyze(output, exit_code=1)
        self.assertFalse(diag.passed)
        self.assertEqual(diag.error_type, "ZeroDivisionError")
        self.assertIn("ZeroDivisionError", diag.failure_summary)
        self.assertIn("test_div_zero (tests.unit.test_calc.TestCalc.test_div_zero)", diag.failing_tests)
        self.assertIn("Division by zero", diag.hypothesized_cause)
        self.assertIn("defensive checks", diag.recommended_fix_direction)
        self.assertTrue(any("src/calc.py:12" in loc for loc in diag.file_locations))

    def test_timeout_failure(self):
        output = "Test Execution Timed Out after 30.0 seconds."
        diag = DiagnosticEngine.analyze(output)
        self.assertFalse(diag.passed)
        self.assertEqual(diag.error_type, "TimeoutError")
        self.assertIn("timed out", diag.failure_summary)
        self.assertIn("exceeded the configured execution timeout", diag.hypothesized_cause)

    def test_syntax_error_failure(self):
        output = (
            "Traceback (most recent call last):\n"
            '  File "src/calc.py", line 10\n'
            "    def broken_syntax(:\n"
            "                      ^\n"
            "SyntaxError: invalid syntax"
        )
        diag = DiagnosticEngine.analyze(output, exit_code=1)
        self.assertFalse(diag.passed)
        self.assertEqual(diag.error_type, "SyntaxError")
        self.assertIn("SyntaxError", diag.failure_summary)
        self.assertIn("syntax or indentation flaw", diag.hypothesized_cause)


if __name__ == "__main__":
    unittest.main()
