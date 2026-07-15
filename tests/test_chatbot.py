from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from portfolio_chatbot.chatbot import ChatTurn, PortfolioChatbot
from portfolio_chatbot.config import ChatSettings, IngestionSettings
from portfolio_chatbot.retrieval import RetrievedChunk


def _chunk(text: str, source: str) -> RetrievedChunk:
    return RetrievedChunk(text=text, metadata={"source": source}, distance=None)


def _chatbot_without_api_key() -> PortfolioChatbot:
    chatbot = PortfolioChatbot.__new__(PortfolioChatbot)
    chatbot.chat_settings = ChatSettings(
        chat_model="test-chat-model",
        retrieval_top_k=2,
        max_history_turns=4,
    )
    chatbot.ingestion_settings = IngestionSettings(
        docs_path=Path("docs"),
        embedding_model="test-embedding-model",
        chunk_size=500,
        chunk_overlap=50,
        embedding_batch_size=64,
    )
    return chatbot


class ChatbotPromptTests(unittest.TestCase):
    def test_prompt_separates_portfolio_and_uploaded_context(self) -> None:
        chatbot = _chatbot_without_api_key()
        prompt = chatbot._build_user_prompt(
            "Does this JD suit Prince?",
            history=[ChatTurn(question="Previous question", answer="Previous answer")],
            chunks=[_chunk("Prince has Python and RAG experience.", "resume.md")],
            uploaded_chunks=[_chunk("JD requires Python and vector search.", "Uploaded file: jd.txt")],
        )

        self.assertIn("Prince portfolio context:", prompt)
        self.assertIn("Prince has Python and RAG experience.", prompt)
        self.assertIn("Uploaded file context:", prompt)
        self.assertIn("JD requires Python and vector search.", prompt)
        self.assertIn("Previous question", prompt)
        self.assertIn("Does this JD suit Prince?", prompt)

    def test_retrieval_query_includes_uploaded_excerpt(self) -> None:
        chatbot = _chatbot_without_api_key()
        query = chatbot._build_retrieval_query(
            "Give interview questions.",
            history=[],
            uploaded_chunks=[_chunk("JD requires FastAPI and OpenAI.", "Uploaded file: jd.txt")],
        )

        self.assertIn("Uploaded file excerpt:", query)
        self.assertIn("JD requires FastAPI and OpenAI.", query)
        self.assertIn("Give interview questions.", query)

    def test_ask_combines_retrieved_and_uploaded_sources(self) -> None:
        chatbot = _chatbot_without_api_key()
        fake_completions = _FakeCompletions()
        chatbot.client = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))

        uploaded_chunk = _chunk("JD asks for Python and OpenAI.", "Uploaded file: jd.txt")
        portfolio_chunk = _chunk("Prince has Python and OpenAI project experience.", "resume.md")

        with patch(
            "portfolio_chatbot.chatbot.retrieve_similar_chunks",
            return_value=[portfolio_chunk],
        ) as retrieve_mock:
            response = chatbot.ask(
                "Does this JD fit Prince?",
                history=[],
                uploaded_chunks=[uploaded_chunk],
            )

        self.assertEqual(response.answer, "Grounded answer")
        self.assertEqual(response.sources, ["resume.md", "Uploaded file: jd.txt"])
        self.assertEqual(response.retrieved_chunks, [portfolio_chunk, uploaded_chunk])
        retrieve_mock.assert_called_once()

        prompt = fake_completions.calls[0]["messages"][1]["content"]
        self.assertIn("Prince portfolio context:", prompt)
        self.assertIn("Uploaded file context:", prompt)
        self.assertIn("JD asks for Python and OpenAI.", prompt)


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="Grounded answer")),
            ]
        )


if __name__ == "__main__":
    unittest.main()
