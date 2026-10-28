"""Unit tests for embedding providers (Mock and Gemini)."""

import math
import unittest
from unittest.mock import MagicMock, patch

from codebase_assistant.embeddings.gemini import GeminiEmbeddingProvider
from codebase_assistant.embeddings.mock import MockEmbeddingProvider


class TestEmbeddingProviders(unittest.TestCase):
    """Test suite for MockEmbeddingProvider and GeminiEmbeddingProvider."""

    def test_mock_embedding_provider_deterministic(self):
        provider = MockEmbeddingProvider(dimension=256)
        self.assertEqual(provider.dimension, 256)
        self.assertEqual(provider.model_name, "mock-embedding-v1")

        text = "def calculate_statistics(data: list) -> dict:"
        vec1 = provider.embed_text(text)
        vec2 = provider.embed_text(text)

        self.assertEqual(len(vec1), 256)
        self.assertEqual(vec1, vec2)

        # Verify unit length (L2 norm == 1.0)
        norm = math.sqrt(sum(x * x for x in vec1))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_mock_embedding_provider_token_sensitivity(self):
        provider = MockEmbeddingProvider(dimension=512)

        doc_rag = "class RAGPipeline: retrieves context and calls LLM"
        doc_rag_similar = "def run_rag_pipeline(retriever, llm): passes retrieved context"
        doc_unrelated = "import matplotlib.pyplot as plt\nplt.plot([1, 2, 3])"

        v_rag = provider.embed_text(doc_rag)
        v_similar = provider.embed_text(doc_rag_similar)
        v_unrelated = provider.embed_text(doc_unrelated)

        # Dot product == cosine similarity for unit vectors
        sim_similar = sum(a * b for a, b in zip(v_rag, v_similar))
        sim_unrelated = sum(a * b for a, b in zip(v_rag, v_unrelated))

        self.assertGreater(sim_similar, sim_unrelated)
        self.assertGreater(sim_similar, 0.2)

    def test_mock_embedding_batch(self):
        provider = MockEmbeddingProvider(dimension=128)
        texts = ["alpha", "beta", "gamma"]
        batch_vecs = provider.embed_batch(texts)

        self.assertEqual(len(batch_vecs), 3)
        for i, t in enumerate(texts):
            self.assertEqual(batch_vecs[i], provider.embed_text(t))

    def test_mock_embedding_empty_text(self):
        provider = MockEmbeddingProvider(dimension=64)
        vec = provider.embed_text("")
        self.assertEqual(len(vec), 64)
        norm = math.sqrt(sum(x * x for x in vec))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_gemini_embedding_provider_requires_key(self):
        with patch("codebase_assistant.embeddings.gemini.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                gemini_embedding_api_key=None,
                gemini_api_key=None,
                gemini_embedding_model="gemini-embedding-001",
            )
            provider = GeminiEmbeddingProvider(api_key=None)
            with self.assertRaises(ValueError) as ctx:
                provider.embed_text("sample")
            self.assertIn("GEMINI_API_KEY is required", str(ctx.exception))

    def test_gemini_embedding_provider_mocked_client(self):
        mock_emb_1 = MagicMock()
        mock_emb_1.values = [0.6, 0.8]  # unit norm == 1.0

        mock_response = MagicMock()
        mock_response.embeddings = [mock_emb_1]

        mock_client = MagicMock()
        mock_client.models.embed_content.return_value = mock_response

        with patch("google.genai.Client", return_value=mock_client):
            provider = GeminiEmbeddingProvider(
                api_key="fake-test-key",
                dimension=2,
            )
            # Inject client
            provider._client = mock_client

            vec = provider.embed_text("test text")
            self.assertEqual(len(vec), 2)
            self.assertAlmostEqual(vec[0], 0.6)
            self.assertAlmostEqual(vec[1], 0.8)

            mock_client.models.embed_content.assert_called_once()


if __name__ == "__main__":
    unittest.main()
