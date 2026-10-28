"""Unit tests for repository scanner and path filtering."""

from pathlib import Path
import tempfile
import unittest

from codebase_assistant.ingestion.scanner import RepositoryScanner


class TestRepositoryScanner(unittest.TestCase):
    """Test path filtering, ignore rules, and secret exclusion."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_secret_file_detection(self):
        scanner = RepositoryScanner(self.root)

        self.assertTrue(scanner.is_secret_file(Path(".env")))
        self.assertTrue(scanner.is_secret_file(Path(".env.local")))
        self.assertTrue(scanner.is_secret_file(Path(".env.production")))
        self.assertTrue(scanner.is_secret_file(Path("certs/server.key")))
        self.assertTrue(scanner.is_secret_file(Path("certs/cert.pem")))
        self.assertTrue(scanner.is_secret_file(Path("ssh/id_rsa")))
        self.assertTrue(scanner.is_secret_file(Path("auth/credentials.json")))
        self.assertTrue(scanner.is_secret_file(Path("auth/token.json")))

        # Standard non-secret files
        self.assertFalse(scanner.is_secret_file(Path("src/main.py")))
        self.assertFalse(scanner.is_secret_file(Path("README.md")))
        self.assertFalse(scanner.is_secret_file(Path("pyproject.toml")))

    def test_ignore_patterns(self):
        scanner = RepositoryScanner(self.root)

        self.assertTrue(scanner.is_ignored(Path(".git/config")))
        self.assertTrue(scanner.is_ignored(Path("src/__pycache__/utils.cpython-310.pyc")))
        self.assertTrue(scanner.is_ignored(Path(".pytest_cache/v/cache/nodeids")))
        self.assertTrue(scanner.is_ignored(Path(".venv/bin/python")))
        self.assertTrue(scanner.is_ignored(Path("deepeval_venv/bin/pytest")))
        self.assertTrue(scanner.is_ignored(Path("dist/package-0.1.0.tar.gz")))
        self.assertTrue(scanner.is_ignored(Path("app.log")))

        # Valid source files
        self.assertFalse(scanner.is_ignored(Path("src/pipeline.py")))
        self.assertFalse(scanner.is_ignored(Path("tests/test_pipeline.py")))

    def test_detect_language(self):
        scanner = RepositoryScanner(self.root)

        self.assertEqual(scanner.detect_language(Path("src/code.py")), "python")
        self.assertEqual(scanner.detect_language(Path("docs/spec.md")), "markdown")
        self.assertEqual(scanner.detect_language(Path("pyproject.toml")), "toml")
        self.assertEqual(scanner.detect_language(Path("data.json")), "json")
        self.assertEqual(scanner.detect_language(Path("Makefile")), "makefile")
        self.assertIsNone(scanner.detect_language(Path("image.png")))

    def test_scan_filters_secrets_and_extracts_files(self):
        # Create sample files
        (self.root / "src").mkdir(parents=True)
        (self.root / "src" / "app.py").write_text("print('hello')", encoding="utf-8")
        (self.root / "README.md").write_text("# Project", encoding="utf-8")
        (self.root / ".env").write_text("SECRET_KEY=12345", encoding="utf-8")
        (self.root / ".env.local").write_text("API_KEY=xyz", encoding="utf-8")
        (self.root / "__pycache__").mkdir()
        (self.root / "__pycache__" / "app.pyc").write_bytes(b"\x00\x01\x02")

        scanner = RepositoryScanner(self.root)
        eligible, revision, excluded = scanner.scan()

        eligible_posix = [p.as_posix() for p in eligible]
        self.assertIn("src/app.py", eligible_posix)
        self.assertIn("README.md", eligible_posix)

        # Ensure secrets and caches are NEVER eligible
        self.assertNotIn(".env", eligible_posix)
        self.assertNotIn(".env.local", eligible_posix)
        self.assertNotIn("__pycache__/app.pyc", eligible_posix)

        # Check exclusion log
        secret_exclusions = [e for e in excluded if "[SECRET]" in e]
        self.assertTrue(any(".env" in e for e in secret_exclusions))


if __name__ == "__main__":
    unittest.main()
