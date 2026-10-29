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

    def test_validate_ast_syntax_empty_code_rejected(self):
        valid, errors = DiffEngine.validate_ast_syntax("")
        self.assertFalse(valid)
        self.assertIn("Code proposal is empty or contains only whitespace.", errors)

        valid, errors = DiffEngine.validate_ast_syntax("   \n\t  ")
        self.assertFalse(valid)
        self.assertIn("Code proposal is empty or contains only whitespace.", errors)

        # Allow empty flag test
        valid, errors = DiffEngine.validate_ast_syntax("", allow_empty=True)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_apply_diff_context_past_eof_rejected(self):
        original = "line1\n"
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,2 +1,3 @@\n"
            " line1\n"
            "+line1.5\n"
            " line2\n"
        )
        with self.assertRaises(ValueError) as ctx:
            DiffEngine.apply_diff_in_memory(original, diff)
        self.assertIn("occurs past end of original content", str(ctx.exception))

    def test_apply_diff_hunk_line_count_mismatch_rejected(self):
        original = "line1\nline2\nline3\n"
        # Claims old_count=3 but only 2 old lines provided
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,2 @@\n"
            " line1\n"
            "-line2\n"
            "+line2_new\n"
        )
        with self.assertRaises(ValueError) as ctx:
            DiffEngine.apply_diff_in_memory(original, diff)
        self.assertIn("header specifies 3 old lines, but hunk contains 2", str(ctx.exception))

    def test_apply_diff_overlapping_or_out_of_order_hunks_rejected(self):
        original = "line1\nline2\nline3\nline4\nline5\nline6\n"
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -5,2 +5,2 @@\n"
            " line5\n"
            "-line6\n"
            "+line6_new\n"
            "@@ -3,2 +3,2 @@\n"
            " line3\n"
            "-line4\n"
            "+line4_new\n"
        )
        with self.assertRaises(ValueError) as ctx:
            DiffEngine.apply_diff_in_memory(original, diff)
        self.assertIn("overlapping or out-of-order hunk", str(ctx.exception))

    def test_validate_target_path_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_path = Path(tmp_dir) / "repo"
            repo_path.mkdir()
            outside_dir = Path(tmp_dir) / "outside"
            outside_dir.mkdir()
            outside_file = outside_dir / "secret.py"
            outside_file.write_text("SECRET = 1\n", encoding="utf-8")

            symlink_path = repo_path / "link_secret.py"
            try:
                symlink_path.symlink_to(outside_file)
            except (OSError, NotImplementedError):
                raise unittest.SkipTest("Symlinks not supported in this environment")

            with self.assertRaises(ValueError) as ctx:
                DiffEngine.validate_target_path(repo_path, "link_secret.py")
            self.assertIn("resolves outside repository via symlink", str(ctx.exception))

    def test_apply_new_file_diff_mismatched_hunk_count_rejected(self):
        diff = (
            "--- /dev/null\n"
            "+++ b/new.py\n"
            "@@ -0,0 +1,5 @@\n"
            "+line1\n"
            "+line2\n"
        )
        with self.assertRaises(ValueError) as ctx:
            DiffEngine.apply_diff_in_memory("", diff)
        self.assertIn("header specifies 5 new lines, but hunk contains 2", str(ctx.exception))

    def test_apply_new_file_diff_with_context_or_deletion_rejected(self):
        # Case A: Header claims old lines from /dev/null
        diff_old = (
            "--- /dev/null\n"
            "+++ b/new.py\n"
            "@@ -0,1 +1,2 @@\n"
            "-old\n"
            "+new\n"
        )
        with self.assertRaises(ValueError) as ctx:
            DiffEngine.apply_diff_in_memory("", diff_old)
        self.assertIn("header specifies old_count=1 but source is /dev/null", str(ctx.exception))

        # Case B: Header claims old_count=0 but lines contain deletion or context
        diff_del = (
            "--- /dev/null\n"
            "+++ b/new.py\n"
            "@@ -0,0 +1,2 @@\n"
            "-old\n"
            "+new\n"
        )
        with self.assertRaises(ValueError) as ctx:
            DiffEngine.apply_diff_in_memory("", diff_del)
        self.assertIn("cannot delete or match context lines for /dev/null source", str(ctx.exception))

    def test_apply_new_file_diff_missing_hunk_header_rejected(self):
        diff = (
            "--- /dev/null\n"
            "+++ b/new.py\n"
            "+just added line without header\n"
        )
        with self.assertRaises(ValueError) as ctx:
            DiffEngine.apply_diff_in_memory("", diff)
        self.assertTrue(
            "Malformed" in str(ctx.exception),
            f"Expected malformed patch error, got: {ctx.exception}",
        )

    def test_apply_new_file_diff_on_non_empty_original_rejected(self):
        diff = (
            "--- /dev/null\n"
            "+++ b/new.py\n"
            "@@ -0,0 +1,1 @@\n"
            "+line\n"
        )
        with self.assertRaises(ValueError) as ctx:
            DiffEngine.apply_diff_in_memory("existing content\n", diff)
        self.assertIn("Cannot apply new-file patch (--- /dev/null) to non-empty original content", str(ctx.exception))

    def test_apply_insertion_hunk_after_line(self):
        original = "line1\nline2\nline3\n"
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,0 +2,1 @@\n"
            "+inserted_after_line1\n"
        )
        patched = DiffEngine.apply_diff_in_memory(original, diff)
        self.assertEqual(patched, "line1\ninserted_after_line1\nline2\nline3\n")

    def test_apply_insertion_hunk_before_first_line(self):
        original = "line1\nline2\nline3\n"
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -0,0 +1,1 @@\n"
            "+inserted_at_top\n"
        )
        patched = DiffEngine.apply_diff_in_memory(original, diff)
        self.assertEqual(patched, "inserted_at_top\nline1\nline2\nline3\n")

    def test_apply_insertion_hunk_at_eof(self):
        original = "line1\nline2\nline3\n"
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -3,0 +4,1 @@\n"
            "+inserted_at_end\n"
        )
        patched = DiffEngine.apply_diff_in_memory(original, diff)
        self.assertEqual(patched, "line1\nline2\nline3\ninserted_at_end\n")

    def test_apply_multiple_insertion_hunks(self):
        original = "alpha\nbeta\ngamma\n"
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,0 +2,1 @@\n"
            "+after_alpha\n"
            "@@ -2,0 +4,1 @@\n"
            "+after_beta\n"
        )
        patched = DiffEngine.apply_diff_in_memory(original, diff)
        self.assertEqual(patched, "alpha\nafter_alpha\nbeta\nafter_beta\ngamma\n")

    def test_apply_insertion_hunk_position_exceeds_length_rejected(self):
        original = "line1\nline2\n"
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -5,0 +6,1 @@\n"
            "+too_far\n"
        )
        with self.assertRaises(ValueError) as ctx:
            DiffEngine.apply_diff_in_memory(original, diff)
        self.assertIn("Patch hunk insertion position 5 exceeds file length 2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
