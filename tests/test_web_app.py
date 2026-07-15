from __future__ import annotations

import unittest

from fastapi import HTTPException

from portfolio_chatbot.web_app import _parse_history, app


class WebAppTests(unittest.TestCase):
    def test_file_chat_route_is_registered(self) -> None:
        paths = {getattr(route, "path", "") for route in app.routes}

        self.assertIn("/api/chat", paths)
        self.assertIn("/api/chat/file", paths)
        self.assertIn("/api/status", paths)

    def test_parse_history_returns_chat_turns(self) -> None:
        turns = _parse_history('[{"question": "Q1", "answer": "A1"}]')

        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].question, "Q1")
        self.assertEqual(turns[0].answer, "A1")

    def test_parse_history_rejects_invalid_json(self) -> None:
        with self.assertRaises(HTTPException) as context:
            _parse_history("{bad json")

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.detail, "Invalid chat history.")


if __name__ == "__main__":
    unittest.main()
