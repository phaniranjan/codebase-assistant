"""Unit tests for GeminiClient and MockGeminiClient."""

import unittest
from unittest.mock import MagicMock, patch

from codebase_assistant.llm.client import GeminiClient
from codebase_assistant.llm.mock_client import MockGeminiClient


class TestLLMClients(unittest.TestCase):
    """Test suite for GeminiClient and MockGeminiClient."""

    def test_gemini_client_requires_key(self):
        with patch("codebase_assistant.llm.client.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(gemini_api_key=None)
            client = GeminiClient(api_key=None)
            with self.assertRaises(ValueError) as ctx:
                client.generate("test prompt")
            self.assertIn("GEMINI_API_KEY is required", str(ctx.exception))

    def test_gemini_client_mocked_generate(self):
        mock_resp = MagicMock()
        mock_resp.text = "This is a mocked Gemini response."

        mock_genai_client = MagicMock()
        mock_genai_client.models.generate_content.return_value = mock_resp

        client = GeminiClient(api_key="fake-test-key", model_name="gemini-2.5-flash")
        client._client = mock_genai_client

        response = client.generate("Hello Gemini", system_instruction="Be brief")
        self.assertEqual(response.text, "This is a mocked Gemini response.")
        self.assertEqual(response.model_name, "gemini-2.5-flash")
        mock_genai_client.models.generate_content.assert_called_once()

    def test_gemini_client_mocked_streaming(self):
        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk2 = MagicMock()
        chunk2.text = "World"

        mock_genai_client = MagicMock()
        mock_genai_client.models.generate_content_stream.return_value = [chunk1, chunk2]

        client = GeminiClient(api_key="fake-test-key")
        client._client = mock_genai_client

        stream_chunks = list(client.generate_stream("Stream prompt"))
        self.assertEqual(stream_chunks, ["Hello ", "World"])

    def test_mock_gemini_client_synthesizes_grounded_answer(self):
        client = MockGeminiClient()
        prompt = (
            "<UNTRUSTED_REPOSITORY_CONTEXT>\n"
            "### [Reference 1] `src/retriever.py` (lines 15-40, symbol: `BM25Retriever`)\n"
            "```python\ndef search(): pass\n```\n"
            "</UNTRUSTED_REPOSITORY_CONTEXT>\n\n"
            "## Developer Question:\n"
            "How does the retriever work?\n"
        )

        response = client.generate(prompt)
        self.assertIn("src/retriever.py:15-40", response.text)
        self.assertIn("BM25Retriever", response.text)

    def test_mock_gemini_client_missing_context(self):
        client = MockGeminiClient()
        prompt = (
            "<UNTRUSTED_REPOSITORY_CONTEXT>\n"
            "(No relevant repository chunks were found in the index.)\n"
            "</UNTRUSTED_REPOSITORY_CONTEXT>\n\n"
            "## Developer Question:\n"
            "What is the database port?\n"
        )

        response = client.generate(prompt)
        self.assertIn("does not contain sufficient details", response.text)

    def test_mock_gemini_client_streaming(self):
        client = MockGeminiClient(canned_response="A short streamed answer.")
        chunks = list(client.generate_stream("Any prompt"))
        self.assertEqual("".join(chunks), "A short streamed answer.")


if __name__ == "__main__":
    unittest.main()
