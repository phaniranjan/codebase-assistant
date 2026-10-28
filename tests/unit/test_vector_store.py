"""Unit tests for VectorStore."""

import tempfile
import unittest
from pathlib import Path

from codebase_assistant.embeddings.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):
    """Test suite for VectorStore."""

    def test_add_and_search_cosine_similarity(self):
        store = VectorStore(dimension=3)

        # 3 orthogonal or semi-orthogonal vectors
        ids = ["chunk_x", "chunk_y", "chunk_z"]
        vectors = [
            [1.0, 0.0, 0.0],  # along x
            [0.0, 1.0, 0.0],  # along y
            [0.7071, 0.7071, 0.0],  # diagonal x-y
        ]
        metadatas = [
            {"file": "x.py"},
            {"file": "y.py"},
            {"file": "diag.py"},
        ]

        store.add(ids, vectors, metadatas)
        self.assertEqual(store.count(), 3)

        # Query pointing along x
        query = [1.0, 0.0, 0.0]
        results = store.search(query, top_k=2)

        self.assertEqual(len(results), 2)
        # Top hit should be chunk_x with cosine similarity ~ 1.0
        self.assertEqual(results[0][0], "chunk_x")
        self.assertAlmostEqual(results[0][1], 1.0, places=4)

        # Second hit should be chunk_z with cosine similarity ~ 0.7071
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
            store.add(["bad"], [[1.0, 0.0]])  # dimension 2 instead of 3

    def test_atomic_save_and_load(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test_store.json"

            store1 = VectorStore(dimension=4)
            store1.add(
                ["id_a", "id_b"],
                [[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0]],
                [{"meta": "a"}, {"meta": "b"}],
            )
            store1.save(path)
            self.assertTrue(path.is_file())

            # Load into fresh store
            store2 = VectorStore()
            store2.load(path)
            self.assertEqual(store2.count(), 2)
            self.assertEqual(store2.dimension, 4)
            self.assertEqual(store2.get_metadata("id_a")["meta"], "a")

            # Check search on reloaded store
            res = store2.search([1.0, 0.0, 0.0, 0.0], top_k=1)
            self.assertEqual(res[0][0], "id_b")


if __name__ == "__main__":
    unittest.main()
