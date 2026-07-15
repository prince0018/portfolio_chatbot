from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from portfolio_chatbot.chatbot import ChatTurn, PortfolioChatbot
from portfolio_chatbot.chroma_index import describe_chroma_collection
from portfolio_chatbot.config import PROJECT_ROOT
from portfolio_chatbot.uploaded_documents import UploadedDocumentError, load_uploaded_file_chunks


FRONTEND_PATH = PROJECT_ROOT / "frontend"

app = FastAPI(title="Portfolio Chatbot")
app.mount("/assets", StaticFiles(directory=FRONTEND_PATH / "assets"), name="assets")


class ChatMessage(BaseModel):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_PATH / "index.html")


@app.get("/api/status")
def status() -> dict[str, str | int]:
    return describe_chroma_collection()


@app.post("/api/chat")
def chat(request: ChatRequest) -> ChatResponse:
    chatbot = PortfolioChatbot()
    history = _to_chat_turns(request.history)

    try:
        response = chatbot.ask(request.question, history=history)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(answer=response.answer, sources=response.sources)


@app.post("/api/chat/file")
async def chat_with_file(
    request: Request,
    question: str = Query(..., min_length=1),
    history: str = Query(default="[]"),
    file_name: str = Query(default="uploaded-document"),
) -> ChatResponse:
    chatbot = PortfolioChatbot()
    history_turns = _parse_history(history)
    file_content = await request.body()

    try:
        uploaded_chunks = load_uploaded_file_chunks(
            file_name,
            file_content,
            settings=chatbot.ingestion_settings,
        )
        response = chatbot.ask(question, history=history_turns, uploaded_chunks=uploaded_chunks)
    except UploadedDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ChatResponse(answer=response.answer, sources=response.sources)


def _parse_history(history_json: str) -> list[ChatTurn]:
    try:
        raw_history = json.loads(history_json)
        messages = [ChatMessage(**item) for item in raw_history]
    except (TypeError, ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail="Invalid chat history.") from exc

    return _to_chat_turns(messages)


def _to_chat_turns(messages: list[ChatMessage]) -> list[ChatTurn]:
    return [ChatTurn(question=item.question, answer=item.answer) for item in messages]
