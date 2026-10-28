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

    def test_nested_sensitive_files(self):
        scanner = RepositoryScanner(self.root)

        # Deeply nested secret files
        self.assertTrue(scanner.is_secret_file(Path("services/auth/deep/nested/.env.staging")))
        self.assertTrue(scanner.is_secret_file(Path("infra/security/keys/id_ed25519")))
        self.assertTrue(scanner.is_secret_file(Path("config/secrets.json")))
        self.assertTrue(scanner.is_secret_file(Path("deploy/secrets.yaml")))
        self.assertTrue(scanner.is_secret_file(Path("credentials/app.p12")))
        self.assertTrue(scanner.is_secret_file(Path("home/.netrc")))

    def test_content_based_secret_detection(self):
        # File has non-secret name but contains private key header
        cert_file = self.root / "src" / "crypto_data.txt"
        cert_file.parent.mkdir(parents=True, exist_ok=True)
        cert_file.write_text(
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0m...\n-----END RSA PRIVATE KEY-----\n",
            encoding="utf-8",
        )

        scanner = RepositoryScanner(self.root)
        self.assertTrue(scanner.contains_secret_content(cert_file))

        eligible, _, excluded = scanner.scan()
        eligible_posix = [p.as_posix() for p in eligible]
        self.assertNotIn("src/crypto_data.txt", eligible_posix)
        self.assertTrue(any("[SECRET_CONTENT]" in e for e in excluded))

    def test_symlink_pointing_outside_repo_is_excluded(self):
        # Create external file outside repository
        outside_dir = tempfile.TemporaryDirectory()
        outside_file = Path(outside_dir.name) / "external_secret.py"
        outside_file.write_text("API_SECRET = 'outside'\n", encoding="utf-8")

        symlink_path = self.root / "external_link.py"
        try:
            symlink_path.symlink_to(outside_file)
        except OSError:
            # On Windows without developer mode symlink privilege, skip symlink test
            outside_dir.cleanup()
            return

        scanner = RepositoryScanner(self.root)
        eligible, _, excluded = scanner.scan()
        eligible_posix = [p.as_posix() for p in eligible]

        self.assertNotIn("external_link.py", eligible_posix)
        self.assertTrue(any("[SYMLINK_OUTSIDE]" in e for e in excluded))
        outside_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
