"""Security policy engine enforcing command whitelisting, workspace containment, and safe execution."""

import os
from pathlib import Path
import re
import shlex
from typing import List, Tuple

from codebase_assistant.generation.diff_engine import DiffEngine


class SecurityPolicyError(ValueError):
    """Raised when an operation violates security policy or sandbox boundaries."""
    pass


class PolicyEngine:
    """Enforces execution boundaries, whitelisted test commands, and path safety."""

    # Allowed test execution binary patterns
    ALLOWED_COMMAND_PREFIXES = (
        ("pytest",),
        ("python", "-m", "unittest"),
        ("python3", "-m", "unittest"),
    )

    # Disallowed shell operators and command injection tokens
    DANGEROUS_SHELL_TOKENS = {
        ";", "&&", "||", "|", "&", ">", ">>", "<", "<<", "`", "$(",
        "\n", "\r",
    }

    # Disallowed binary executions within command arguments
    DISALLOWED_BINARIES = {
        "rm", "del", "erase", "format", "sh", "bash", "zsh", "cmd", "powershell",
        "pwsh", "curl", "wget", "net", "ssh", "scp", "nc", "ncat", "telnet",
        "sudo", "su", "chmod", "chown", "mkfs", "dd",
    }

    def __init__(self, repository_path: Path | str) -> None:
        self.repository_path = Path(repository_path).resolve()

    def validate_file_access(self, file_path: str | Path) -> Path:
        """Validate that a target file path resolves within the repository root without escaping."""
        try:
            return DiffEngine.validate_target_path(self.repository_path, str(file_path))
        except ValueError as e:
            raise SecurityPolicyError(f"Workspace boundary violation: {str(e)}") from e

    def validate_test_command(self, command: str) -> List[str]:
        """Validate that a test execution command conforms to the strict allowlist.

        Returns split argument list if permitted; raises SecurityPolicyError otherwise.
        """
        if not command or not command.strip():
            raise SecurityPolicyError("Empty command is not permitted.")

        # 1. Reject dangerous shell operators and chaining
        for token in self.DANGEROUS_SHELL_TOKENS:
            if token in command:
                raise SecurityPolicyError(
                    f"Command contains prohibited shell meta-character '{token}'. "
                    f"Shell chaining, redirection, and subshells are strictly disallowed."
                )

        # 2. Parse command arguments safely
        try:
            # On Windows, posix=False helps preserve Windows backslashes in paths
            use_posix = os.name != "nt"
            args = shlex.split(command, posix=use_posix)
        except Exception as e:
            raise SecurityPolicyError(f"Failed to safely parse command string: {e}") from e

        if not args:
            raise SecurityPolicyError("Empty command arguments.")

        # 3. Check for disallowed commands/binaries in args
        for arg in args:
            base_arg = Path(arg).name.lower()
            if base_arg in self.DISALLOWED_BINARIES:
                raise SecurityPolicyError(f"Command references prohibited binary '{arg}'.")

        # 4. Enforce allowed command prefixes
        matched_prefix = False
        lower_args = [a.lower() for a in args]

        for prefix in self.ALLOWED_COMMAND_PREFIXES:
            prefix_len = len(prefix)
            if len(lower_args) >= prefix_len and tuple(lower_args[:prefix_len]) == prefix:
                matched_prefix = True
                break

        if not matched_prefix:
            allowed_desc = [" ".join(p) for p in self.ALLOWED_COMMAND_PREFIXES]
            raise SecurityPolicyError(
                f"Command '{args[0]}' is not an authorized test runner. "
                f"Permitted test commands must begin with: {', '.join(allowed_desc)}"
            )

        # 5. Check remaining path arguments for directory traversal
        for arg in args:
            if ".." in Path(arg).parts:
                raise SecurityPolicyError(f"Path traversal detected in command argument: '{arg}'.")

        return args

    def validate_patch_targets(self, unified_diff: str) -> List[Path]:
        """Inspect unified diff headers and ensure all modified target files reside inside workspace."""
        if not unified_diff or not unified_diff.strip():
            raise SecurityPolicyError("Diff payload is empty.")

        target_paths = []
        for line in unified_diff.splitlines():
            if line.startswith("+++ "):
                raw_path = line[4:].strip()
                # Strip git prefix b/ if present
                clean_path = raw_path[2:] if raw_path.startswith("b/") else raw_path
                if clean_path and clean_path != "/dev/null":
                    validated_path = self.validate_file_access(clean_path)
                    target_paths.append(validated_path)

        if not target_paths:
            raise SecurityPolicyError("Diff does not specify any target file to modify or create.")

        return target_paths
