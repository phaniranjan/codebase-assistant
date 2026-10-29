"""Unit tests for DiffEngine: unified diff generation, in-memory patch application, path validation, and AST checking."""

from pathlib import Path
import tempfile
import unittest

from codebase_assistant.generation.diff_engine import DiffEngine


class TestDiffEngine(unittest.TestCase):
    """Test suite verifying DiffEngine operations and safety guardrails."""

    def test_generate_and_apply_unified_diff_roundtrip(self):
        original = (
            "def calculate(a, b):\n"
            "    # Old implementation\n"
            "    return a + b\n"
        )
        proposed = (
            "def calculate(a, b):\n"
            "    # New improved implementation\n"
            "    if a is None or b is None:\n"
            "        raise ValueError('Inputs must not be None')\n"
            "    return a + b\n"
        )

        diff = DiffEngine.generate_unified_diff(original, proposed, "math/calc.py")
        self.assertIn("--- a/math/calc.py", diff)
        self.assertIn("+++ b/math/calc.py", diff)
        self.assertIn("+    if a is None or b is None:", diff)

        # Apply diff in memory without disk access
        patched = DiffEngine.apply_diff_in_memory(original, diff)
        self.assertEqual(patched, proposed)

    def test_apply_new_file_diff(self):
        original = ""
        proposed = "class NewFeature:\n    pass\n"
        diff = DiffEngine.generate_unified_diff(original, proposed, "new_module.py", is_new_file=True)
        self.assertIn("--- /dev/null", diff)
        self.assertIn("+++ b/new_module.py", diff)

        patched = DiffEngine.apply_diff_in_memory(original, diff)
        self.assertEqual(patched, proposed)

    def test_apply_diff_context_mismatch_raises_error(self):
        original = "x = 10\ny = 20\n"
        # Diff expecting "x = 99" instead of "x = 10"
        malformed_diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-x = 99\n"
            "+x = 100\n"
            " y = 20\n"
        )
        with self.assertRaises(ValueError) as ctx:
            DiffEngine.apply_diff_in_memory(original, malformed_diff)
        self.assertIn("Patch deletion mismatch", str(ctx.exception))

    def test_validate_target_path_workspace_containment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_path = Path(tmp_dir)

            # Valid paths within repository
            valid1 = DiffEngine.validate_target_path(repo_path, "src/main.py")
            self.assertEqual(valid1, (repo_path / "src/main.py").resolve())

            valid2 = DiffEngine.validate_target_path(repo_path, "./tests/test_main.py")
            self.assertEqual(valid2, (repo_path / "tests/test_main.py").resolve())

            # Path traversal attempts
            with self.assertRaises(ValueError) as ctx:
                DiffEngine.validate_target_path(repo_path, "../outside.py")
            self.assertIn("Path traversal detected", str(ctx.exception))

            with self.assertRaises(ValueError) as ctx:
                DiffEngine.validate_target_path(repo_path, "src/../../outside.py")
            self.assertIn("Path traversal detected", str(ctx.exception))

    def test_validate_ast_syntax(self):
        valid_code = "def foo():\n    return 42\n"
        valid, errors = DiffEngine.validate_ast_syntax(valid_code)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

        invalid_code = "def broken(\n    return 42"
        valid, errors = DiffEngine.validate_ast_syntax(invalid_code)
        self.assertFalse(valid)
        self.assertEqual(len(errors), 1)
        self.assertIn("SyntaxError", errors[0])


if __name__ == "__main__":
    unittest.main()
