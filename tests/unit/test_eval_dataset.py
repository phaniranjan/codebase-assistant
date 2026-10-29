"""Unit tests for GoldenDataset schema, path normalization, and repository validation."""

from pathlib import Path
import tempfile
import unittest

from codebase_assistant.evaluation.dataset import (
    EvalCategory,
    GoldenDataset,
    GoldenTestCase,
    normalize_repo_path,
)


class TestGoldenDataset(unittest.TestCase):
    """Test suite for GoldenDataset and GoldenTestCase."""

    def test_normalize_repo_path(self):
        self.assertEqual(normalize_repo_path("src\\module\\file.py"), "src/module/file.py")
        self.assertEqual(normalize_repo_path("./src/module/file.py"), "src/module/file.py")
        self.assertEqual(normalize_repo_path("/src/module/file.py"), "src/module/file.py")
        self.assertEqual(normalize_repo_path("  src/module/file.py  "), "src/module/file.py")

    def test_load_golden_dataset_from_default_file(self):
        dataset_path = Path("data/golden_dataset.json")
        self.assertTrue(dataset_path.is_file(), f"Dataset file missing at {dataset_path.resolve()}")

        dataset = GoldenDataset.load_from_file(dataset_path)
        self.assertEqual(dataset.version, "1.0.0")
        self.assertEqual(dataset.target_repository, "ai-evaluation")
        self.assertEqual(len(dataset.cases), 25)

        # Check category distribution
        categories = {c.category for c in dataset.cases}
        self.assertEqual(len(categories), len(EvalCategory))

        for cat in EvalCategory:
            cat_cases = dataset.filter_cases(category=cat)
            self.assertGreaterEqual(len(cat_cases), 3, f"Category {cat.value} should have >= 3 cases")

    def test_filter_cases(self):
        dataset_path = Path("data/golden_dataset.json")
        dataset = GoldenDataset.load_from_file(dataset_path)

        unanswerable = dataset.filter_cases(is_answerable=False)
        self.assertEqual(len(unanswerable), 4)

        injections = dataset.filter_cases(contains_prompt_injection=True)
        self.assertEqual(len(injections), 3)

    def test_validate_against_actual_ai_eval_repo(self):
        candidates = [
            Path("../ai_evaluation").resolve(),
            Path("../../ai_evaluation").resolve(),
            Path("/home/phaniranjan/AI/ai_evaluation").resolve(),
            Path(r"\\wsl.localhost\Ubuntu-26.04\home\phaniranjan\AI\ai_evaluation").resolve(),
        ]
        ai_eval_dir = None
        for cand in candidates:
            if cand.is_dir() and (cand / "src" / "ai_evaluation").is_dir():
                ai_eval_dir = cand
                break

        if ai_eval_dir is None:
            self.skipTest("ai-evaluation repository not found on local path.")

        dataset_path = Path("data/golden_dataset.json")
        dataset = GoldenDataset.load_from_file(dataset_path)
        errors = dataset.validate_against_repo(ai_eval_dir)

        self.assertEqual(
            errors,
            [],
            f"Grounding validation failed for golden dataset:\n" + "\n".join(errors),
        )

    def test_schema_validation_catches_invalid_file(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tf:
            tf.write("{\"cases\": [{\"id\": \"1\", \"question\": \"q\", \"category\": \"invalid_cat\"}]}")
            tf_path = Path(tf.name)

        try:
            with self.assertRaises(Exception):
                GoldenDataset.load_from_file(tf_path)
        finally:
            if tf_path.is_file():
                tf_path.unlink()


if __name__ == "__main__":
    unittest.main()
