import os
import unittest
from unittest.mock import Mock, patch

import openrouter


class OpenRouterTest(unittest.TestCase):
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("openrouter.httpx.post")
    def test_embeddings_use_model_and_input_type(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {
            "data": [
                {"index": 1, "embedding": [3.0, 4.0]},
                {"index": 0, "embedding": [1.0, 2.0]},
            ]
        }
        post.return_value = response

        with patch.object(openrouter, "OPENROUTER_EMBED_DIM", 2):
            result = openrouter.embed_texts(["first", "second"], input_type="passage")

        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])
        request = post.call_args.kwargs
        self.assertEqual(request["json"]["model"], openrouter.OPENROUTER_EMBED_MODEL)
        self.assertEqual(request["json"]["input_type"], "passage")
        response.raise_for_status.assert_called_once()

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"})
    @patch("openrouter.httpx.post")
    def test_chat_completion_uses_free_router(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": "  answer  "}}]
        }
        post.return_value = response

        result = openrouter.chat_completion([{"role": "user", "content": "question"}])

        self.assertEqual(result, "answer")
        self.assertEqual(
            post.call_args.kwargs["json"]["model"], openrouter.OPENROUTER_CHAT_MODEL
        )
        response.raise_for_status.assert_called_once()


if __name__ == "__main__":
    unittest.main()
