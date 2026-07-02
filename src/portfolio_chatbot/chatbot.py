from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI, OpenAIError

from portfolio_chatbot.config import (
    ChatSettings,
    get_chat_settings,
    get_ingestion_settings,
    get_openai_api_key,
)
from portfolio_chatbot.retrieval import RetrievedChunk, retrieve_similar_chunks


SYSTEM_PROMPT = """You are Prince's portfolio chatbot.
Use Prince portfolio context for facts about Prince.
Use uploaded file context only as user-provided document context, such as a job description.
When asked about job fit, compare the uploaded file requirements with Prince portfolio context and call out matches, gaps, and preparation advice.
When asked for interview preparation, create questions grounded in the uploaded file and Prince portfolio context when relevant.
If the available context does not contain the answer, say that the documents do not include that information.
Do not invent projects, skills, experience, dates, education, or contact details.
Keep answers concise, helpful, and professional."""


@dataclass(frozen=True)
class ChatTurn:
    question: str
    answer: str


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    sources: list[str]
    retrieved_chunks: list[RetrievedChunk]


class PortfolioChatbot:
    def __init__(self, chat_settings: ChatSettings | None = None) -> None:
        self.chat_settings = chat_settings or get_chat_settings()
        self.ingestion_settings = get_ingestion_settings()
        self.client = OpenAI(api_key=get_openai_api_key())

    def ask(
        self,
        question: str,
        history: list[ChatTurn] | None = None,
        uploaded_chunks: list[RetrievedChunk] | None = None,
    ) -> ChatResponse:
        history = history or []
        uploaded_chunks = uploaded_chunks or []
        retrieval_query = self._build_retrieval_query(question, history, uploaded_chunks)
        chunks = retrieve_similar_chunks(
            retrieval_query,
            embedding_model=self.ingestion_settings.embedding_model,
            top_k=self.chat_settings.retrieval_top_k,
        )

        if not chunks and not uploaded_chunks:
            return ChatResponse(
                answer=(
                    "I do not have any indexed document chunks yet. "
                    "Please ingest your documents first, then ask again."
                ),
                sources=[],
                retrieved_chunks=[],
            )

        try:
            response = self.client.chat.completions.create(
                model=self.chat_settings.chat_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": self._build_user_prompt(
                            question,
                            history,
                            chunks,
                            uploaded_chunks,
                        ),
                    },
                ],
            )
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI chat request failed: {exc}") from exc

        answer = response.choices[0].message.content or ""

        return ChatResponse(
            answer=answer.strip(),
            sources=_unique_sources([*chunks, *uploaded_chunks]),
            retrieved_chunks=[*chunks, *uploaded_chunks],
        )

    def _build_retrieval_query(
        self,
        question: str,
        history: list[ChatTurn],
        uploaded_chunks: list[RetrievedChunk],
    ) -> str:
        recent_history = history[-self.chat_settings.max_history_turns :]
        uploaded_excerpt = _format_uploaded_retrieval_excerpt(uploaded_chunks)
        if not recent_history:
            if uploaded_excerpt:
                return f"Uploaded file excerpt:\n{uploaded_excerpt}\n\nUser question:\n{question}"
            return question

        history_text = "\n".join(
            f"User: {turn.question}\nAssistant: {turn.answer}" for turn in recent_history
        )
        if uploaded_excerpt:
            return f"{history_text}\nUploaded file excerpt:\n{uploaded_excerpt}\nUser: {question}"
        return f"{history_text}\nUser: {question}"

    def _build_user_prompt(
        self,
        question: str,
        history: list[ChatTurn],
        chunks: list[RetrievedChunk],
        uploaded_chunks: list[RetrievedChunk],
    ) -> str:
        recent_history = history[-self.chat_settings.max_history_turns :]
        conversation = _format_history(recent_history)
        prince_context = _format_context(chunks) or "No Prince portfolio context was retrieved."
        uploaded_context = _format_context(uploaded_chunks) or "No uploaded file was provided."

        return f"""Prince portfolio context:
{prince_context}

Uploaded file context:
{uploaded_context}

Recent conversation:
{conversation}

Question:
{question}

Answer using only the Prince portfolio context and uploaded file context. If the answer is not in the context, say so."""


def _format_history(history: list[ChatTurn]) -> str:
    if not history:
        return "No previous conversation."

    return "\n".join(f"User: {turn.question}\nAssistant: {turn.answer}" for turn in history)


def _format_context(chunks: list[RetrievedChunk]) -> str:
    formatted_chunks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        source = _source_label(chunk)
        formatted_chunks.append(f"[Context {index}: {source}]\n{chunk.text}")

    return "\n\n".join(formatted_chunks)


def _format_uploaded_retrieval_excerpt(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ""

    excerpt = "\n\n".join(chunk.text for chunk in chunks)
    return excerpt[:4000]


def _unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    sources: list[str] = []

    for chunk in chunks:
        source = _source_label(chunk)
        if source not in sources:
            sources.append(source)

    return sources


def _source_label(chunk: RetrievedChunk) -> str:
    source = chunk.metadata.get("source") or chunk.metadata.get("file_name") or "unknown source"
    page = chunk.metadata.get("page")

    if page:
        return f"{source}, page {page}"

    return str(source)
