from __future__ import annotations

import tempfile
from pathlib import Path

from portfolio_chatbot.chunking import TextChunk, split_documents_into_chunks
from portfolio_chatbot.config import IngestionSettings
from portfolio_chatbot.document_loaders import SUPPORTED_EXTENSIONS, load_document
from portfolio_chatbot.retrieval import RetrievedChunk


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_UPLOADED_CONTEXT_CHARS = 14_000


class UploadedDocumentError(ValueError):
    pass


def load_uploaded_file_chunks(
    file_name: str,
    content: bytes,
    *,
    settings: IngestionSettings,
) -> list[RetrievedChunk]:
    safe_file_name = Path(file_name).name.strip() or "uploaded-document"
    suffix = Path(safe_file_name).suffix.lower()

    if not content:
        raise UploadedDocumentError("The uploaded file is empty.")

    if len(content) > MAX_UPLOAD_BYTES:
        max_size_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise UploadedDocumentError(f"Upload a file smaller than {max_size_mb} MB.")

    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UploadedDocumentError(f"Unsupported file type. Please upload one of: {supported}.")

    with tempfile.TemporaryDirectory(prefix="portfolio-chatbot-upload-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        uploaded_path = temp_dir / safe_file_name
        uploaded_path.write_bytes(content)
        documents = load_document(uploaded_path, temp_dir)

    if not documents:
        raise UploadedDocumentError("I could not extract readable text from that file.")

    chunks = split_documents_into_chunks(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    selected_chunks = _limit_chunks_to_context(chunks)

    return [
        RetrievedChunk(
            text=chunk.text,
            metadata={
                **chunk.metadata,
                "source": f"Uploaded file: {safe_file_name}",
                "file_name": safe_file_name,
                "uploaded": True,
            },
            distance=None,
        )
        for chunk in selected_chunks
    ]


def _limit_chunks_to_context(chunks: list[TextChunk]) -> list[TextChunk]:
    selected: list[TextChunk] = []
    used_chars = 0

    for chunk in chunks:
        next_size = len(chunk.text)
        if selected and used_chars + next_size > MAX_UPLOADED_CONTEXT_CHARS:
            break

        selected.append(chunk)
        used_chars += next_size

    return selected
