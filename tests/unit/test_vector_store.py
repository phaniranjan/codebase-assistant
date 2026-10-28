"""Unit tests for VectorStore testing all execution tiers and persistence."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from codebase_assistant.embeddings.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """Test suite for VectorStore across all three execution tiers."""

    def test_add_and_search_cosine_similarity(self):
        store = VectorStore(dimension=3)

        ids = ["chunk_x", "chunk_y", "chunk_z"]
        vectors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7071, 0.7071, 0.0],
        ]
        metadatas = [
            {"file": "x.py"},
            {"file": "y.py"},
            {"file": "diag.py"},
        ]

        store.add(ids, vectors, metadatas)
        self.assertEqual(store.count(), 3)

        query = [1.0, 0.0, 0.0]
        results = store.search(query, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "chunk_x")
        self.assertAlmostEqual(results[0][1], 1.0, places=4)
        self.assertEqual(results[1][0], "chunk_z")
        self.assertAlmostEqual(results[1][1], 0.7071, places=3)

    def test_remove_vectors(self):
        store = VectorStore(dimension=2)
        store.add(["c1", "c2"], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(store.count(), 2)

        store.remove(["c1"])
        self.assertEqual(store.count(), 1)
        self.assertIsNone(store.get_vector("c1"))
        self.assertIsNotNone(store.get_vector("c2"))

    def test_dimension_mismatch_raises_error(self):
        store = VectorStore(dimension=3)
        with self.assertRaises(ValueError):
            store.add(["bad"], [[1.0, 0.0]])

    def test_persistence_preserves_vectors_ids_metadata_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test_store.json"
            fp = {
                "provider": "GeminiEmbeddingProvider",
                "model_name": "gemini-embedding-001",
                "dimension": 4,
                "normalized": True,
            }

            store1 = VectorStore(dimension=4, fingerprint=fp)
            store1.add(
                ["id_a", "id_b"],
                [[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0]],
                [{"meta": "a", "line": 10}, {"meta": "b", "line": 20}],
            )
            store1.save(path)
            self.assertTrue(path.is_file())

            # Load into fresh store
            store2 = VectorStore()
            store2.load(path)
            self.assertEqual(store2.count(), 2)
            self.assertEqual(store2.dimension, 4)
            self.assertEqual(store2.fingerprint, fp)
            self.assertEqual(store2.get_metadata("id_a")["line"], 10)
            self.assertEqual(store2.get_metadata("id_b")["meta"], "b")

            # Verify vectors were restored accurately
            vec_b = store2.get_vector("id_b")
            self.assertIsNotNone(vec_b)
            self.assertAlmostEqual(vec_b[0], 1.0)
            self.assertAlmostEqual(vec_b[1], 0.0)

            # Check search on reloaded store
            res = store2.search([1.0, 0.0, 0.0, 0.0], top_k=1)
            self.assertEqual(res[0][0], "id_b")

    def test_tier1_faiss_search_execution(self):
        """Tier 1: Verify FAISS IndexFlatIP dispatch when faiss and numpy are present."""
        store = VectorStore(dimension=2)

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.search.return_value = ([[0.999]], [[0]])
        mock_faiss.IndexFlatIP.return_value = mock_index

        mock_np = MagicMock()
        mock_np.float32 = "float32"
        mock_np.array.side_effect = lambda x, **kwargs: MagicMock(shape=(len(x), 2)) if isinstance(x, list) else MagicMock()

        store._faiss = mock_faiss
        store._numpy = mock_np

        ids = [f"c_{i}" for i in range(25)]
        vecs = [[1.0, 0.0] for _ in range(25)]
        store.add(ids, vecs)

        results = store.search([1.0, 0.0], top_k=1)
        self.assertEqual(len(results), 1)
        mock_faiss.IndexFlatIP.assert_called_once()
        mock_faiss.normalize_L2.assert_called()

    def test_tier2_numpy_search_execution(self):
        """Tier 2: Verify NumPy matrix dot-product dispatch when faiss is absent."""
        store = VectorStore(dimension=2)
        store._faiss = None  # No FAISS

        mock_np = MagicMock()
        mock_np.float32 = "float32"
        mock_np.array.return_value = MagicMock()
        mock_np.dot.return_value = [0.95] * 25
        mock_np.argsort.return_value = list(range(25))[::-1]

        store._numpy = mock_np

        ids = [f"c_{i}" for i in range(25)]
        vecs = [[1.0, 0.0] for _ in range(25)]
        store.add(ids, vecs)

        results = store.search([1.0, 0.0], top_k=2)
        self.assertEqual(len(results), 2)
        mock_np.dot.assert_called_once()

    def test_tier3_pure_python_search_execution(self):
        """Tier 3: Verify Pure-Python fallback when neither FAISS nor NumPy is present."""
        store = VectorStore(dimension=2)
        store._faiss = None
        store._numpy = None

        store.add(["px", "py"], [[1.0, 0.0], [0.0, 1.0]])
        results = store.search([1.0, 0.0], top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "px")
        self.assertAlmostEqual(results[0][1], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
