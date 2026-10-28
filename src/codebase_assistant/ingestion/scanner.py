"""Repository scanner with path filtering, git ignore support, and secret exclusion."""

import fnmatch
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from codebase_assistant.config import get_settings
from codebase_assistant.ingestion.models import SourceRevision

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".md": "markdown",
    ".markdown": "markdown",
    ".toml": "toml",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".txt": "text",
    ".cfg": "ini",
    ".ini": "ini",
}

SPECIAL_FILENAMES = {
    "Makefile": "makefile",
    "Dockerfile": "dockerfile",
    ".editorconfig": "ini",
}


class RepositoryScanner:
    """Scans a repository filesystem, filtering out secrets, build artifacts, and ignored paths."""

    def __init__(
        self,
        repository_path: Path,
        ignore_patterns: Optional[List[str]] = None,
        secret_patterns: Optional[List[str]] = None,
    ) -> None:
        self.repository_path = Path(repository_path).resolve()
        settings = get_settings()
        self.ignore_patterns = list(ignore_patterns or settings.default_ignore_patterns)
        self.secret_patterns = list(secret_patterns or settings.secret_patterns)

        # Load repository .gitignore if it exists
        self.gitignore_patterns = self._load_gitignore()

    def _load_gitignore(self) -> List[str]:
        """Read .gitignore rules from repository root if available."""
        gitignore_path = self.repository_path / ".gitignore"
        patterns: List[str] = []
        if gitignore_path.is_file():
            try:
                with open(gitignore_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            patterns.append(line)
            except Exception as e:
                logger.warning("Could not read .gitignore at %s: %s", gitignore_path, e)
        return patterns

    def is_secret_file(self, rel_path: Path) -> bool:
        """Check if path matches known secret, token, or credential patterns.

        Limits of Secret Detection:
        Matches file names, suffixes, and parent directories against known credential
        patterns (.env, *.key, *.pem, id_rsa, credentials.json, token.json).
        This does NOT perform entropy analysis or arbitrary string searches in code.
        """
        filename = rel_path.name.lower()
        rel_str = rel_path.as_posix().lower()

        for pattern in self.secret_patterns:
            pat_lower = pattern.lower()
            if fnmatch.fnmatch(filename, pat_lower) or fnmatch.fnmatch(rel_str, pat_lower):
                return True
            # Check nested path components
            if any(fnmatch.fnmatch(part.lower(), pat_lower) for part in rel_path.parts):
                return True
            # Check prefix/suffix variants like .env.local, .env.staging
            if pat_lower.startswith(".env") and (filename == ".env" or filename.startswith(".env.")):
                return True
            if pat_lower.endswith(".env") and filename.endswith(".env"):
                return True
        return False

    @staticmethod
    def contains_secret_content(abs_path: Path) -> bool:
        """Inspect file header for standard cryptographic private key blocks.

        Limits of Secret Detection:
        Scans the initial 4KB for PEM private key headers (e.g. '-----BEGIN PRIVATE KEY-----').
        It will not detect inline API keys, high-entropy passwords, or database connection strings
        embedded within code files. Those should be intercepted by pre-commit hooks (e.g., Gitleaks).
        """
        if not abs_path.is_file():
            return False
        try:
            with open(abs_path, "rb") as f:
                head = f.read(4096)
                if b"-----BEGIN " in head and (
                    b"PRIVATE KEY-----" in head
                    or b"RSA PRIVATE KEY-----" in head
                    or b"OPENSSH PRIVATE KEY-----" in head
                    or b"EC PRIVATE KEY-----" in head
                    or b"DSA PRIVATE KEY-----" in head
                    or b"PGP PRIVATE KEY BLOCK-----" in head
                ):
                    return True
        except Exception:
            pass
        return False

    def is_ignored_directory(self, dir_name: str, rel_dir: Path) -> bool:
        """Fast check to prune directories before walking into them."""
        # Top-level/part matching
        fast_ignore_names = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".venv",
            "venv",
            "deepeval_venv",
            "build",
            "dist",
            ".codebase_assistant",
            ".tox",
        }
        if dir_name in fast_ignore_names:
            return True

        if dir_name.endswith(".egg-info"):
            return True

        rel_str = (rel_dir / dir_name).as_posix()
        for pat in self.ignore_patterns + self.gitignore_patterns:
            clean = pat.rstrip("/*")
            if fnmatch.fnmatch(dir_name, clean) or fnmatch.fnmatch(rel_str, clean):
                return True

        return False

    def is_ignored(self, rel_path: Path) -> bool:
        """Check if path matches default ignore patterns or .gitignore rules."""
        rel_str = rel_path.as_posix()
        parts = rel_path.parts

        # Check part-level ignore
        for part in parts:
            for pattern in self.ignore_patterns:
                clean_pat = pattern.rstrip("/*")
                if fnmatch.fnmatch(part, clean_pat):
                    return True

        # Check full relative path pattern
        for pattern in self.ignore_patterns + self.gitignore_patterns:
            clean_pat = pattern.rstrip("/")
            if fnmatch.fnmatch(rel_str, clean_pat) or fnmatch.fnmatch(rel_str, f"{clean_pat}/*"):
                return True
            if any(fnmatch.fnmatch(part, clean_pat) for part in parts):
                return True

        return False

    def detect_language(self, rel_path: Path) -> Optional[str]:
        """Detect file language/type based on extension or special filename."""
        if rel_path.name in SPECIAL_FILENAMES:
            return SPECIAL_FILENAMES[rel_path.name]
        suffix = rel_path.suffix.lower()
        return SUPPORTED_EXTENSIONS.get(suffix)

    def get_source_revision(self) -> SourceRevision:
        """Extract repository revision (Git commit SHA, branch) directly from .git."""
        git_dir = self.repository_path / ".git"
        if not git_dir.exists():
            return SourceRevision(revision_id="no-git")

        try:
            head_file = git_dir / "HEAD"
            if head_file.is_file():
                head_content = head_file.read_text(encoding="utf-8").strip()
                if head_content.startswith("ref:"):
                    ref_path = head_content.split(":", 1)[1].strip()
                    branch_name = ref_path.replace("refs/heads/", "")
                    full_ref = git_dir / ref_path
                    if full_ref.is_file():
                        commit_sha = full_ref.read_text(encoding="utf-8").strip()
                        return SourceRevision(
                            commit_sha=commit_sha,
                            branch=branch_name,
                            revision_id=commit_sha[:8] if commit_sha else branch_name,
                        )
                    # Check packed-refs if branch ref file not yet unpacked
                    packed_refs = git_dir / "packed-refs"
                    if packed_refs.is_file():
                        for line in packed_refs.read_text(encoding="utf-8", errors="ignore").splitlines():
                            parts = line.strip().split()
                            if len(parts) == 2 and parts[1] == ref_path:
                                return SourceRevision(
                                    commit_sha=parts[0],
                                    branch=branch_name,
                                    revision_id=parts[0][:8],
                                )
                    return SourceRevision(branch=branch_name, revision_id=branch_name)
                else:
                    # Detached HEAD
                    return SourceRevision(
                        commit_sha=head_content,
                        revision_id=head_content[:8],
                    )
        except Exception as e:
            logger.debug("Failed to read .git revision directly: %s", e)

        return SourceRevision(revision_id="git-unresolved")

    @staticmethod
    def compute_file_hash(abs_path: Path) -> str:
        """Compute SHA-256 hash of a file's content."""
        hasher = hashlib.sha256()
        with open(abs_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def scan(self) -> Tuple[List[Path], SourceRevision, List[str]]:
        """Scan repository and return (eligible_rel_paths, source_revision, excluded_paths)."""
        if not self.repository_path.is_dir():
            raise FileNotFoundError(f"Repository directory does not exist: {self.repository_path}")

        eligible_files: List[Path] = []
        excluded_files: List[str] = []
        revision = self.get_source_revision()

        repo_str = str(self.repository_path)

        for root, dirs, files in os.walk(repo_str):
            root_path = Path(root)
            try:
                rel_dir = root_path.relative_to(self.repository_path)
            except ValueError:
                continue

            # Prune ignored directories in-place so os.walk NEVER descends into them
            pruned_dirs = []
            retained_dirs = []
            for d in dirs:
                if self.is_ignored_directory(d, rel_dir):
                    pruned_dirs.append(d)
                    excluded_files.append(f"[IGNORED_DIR] {(rel_dir / d).as_posix()}")
                else:
                    retained_dirs.append(d)
            dirs[:] = retained_dirs

            # Check individual files
            for f in files:
                rel_path = rel_dir / f if rel_dir != Path(".") else Path(f)
                abs_path = root_path / f

                # 0. Symlink verification: never traverse symlinks pointing outside repository
                if abs_path.is_symlink():
                    try:
                        resolved = abs_path.resolve()
                        resolved.relative_to(self.repository_path)
                    except (ValueError, RuntimeError):
                        excluded_files.append(f"[SYMLINK_OUTSIDE] {rel_path.as_posix()}")
                        continue
                    if not resolved.exists():
                        excluded_files.append(f"[BROKEN_SYMLINK] {rel_path.as_posix()}")
                        continue

                # 1. Enforce strict secret exclusion (filename & path rules)
                if self.is_secret_file(rel_path):
                    excluded_files.append(f"[SECRET] {rel_path.as_posix()}")
                    continue

                # 2. Enforce content-level secret detection (PEM / private keys)
                if self.contains_secret_content(abs_path):
                    excluded_files.append(f"[SECRET_CONTENT] {rel_path.as_posix()}")
                    continue

                # 3. Enforce ignore patterns
                if self.is_ignored(rel_path):
                    excluded_files.append(f"[IGNORED] {rel_path.as_posix()}")
                    continue

                # 4. Check language support
                lang = self.detect_language(rel_path)
                if not lang:
                    excluded_files.append(f"[UNSUPPORTED] {rel_path.as_posix()}")
                    continue

                eligible_files.append(rel_path)

        eligible_files.sort(key=lambda p: p.as_posix())
        return eligible_files, revision, excluded_files
