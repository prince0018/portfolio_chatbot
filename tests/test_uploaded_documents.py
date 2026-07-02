from __future__ import annotations

import unittest
from pathlib import Path

from portfolio_chatbot.config import IngestionSettings
from portfolio_chatbot.uploaded_documents import UploadedDocumentError, load_uploaded_file_chunks


def _settings() -> IngestionSettings:
    return IngestionSettings(
        docs_path=Path("docs"),
        embedding_model="text-embedding-ada-002",
        chunk_size=500,
        chunk_overlap=50,
        embedding_batch_size=64,
    )


class UploadedDocumentTests(unittest.TestCase):
    def test_load_text_upload_creates_temporary_context_chunks(self) -> None:
        chunks = load_uploaded_file_chunks(
            "../sample-jd.txt",
            b"Role: AI Engineer\nSkills: Python, OpenAI, RAG.\nPrepare interview questions.",
            settings=_settings(),
        )

        self.assertEqual(len(chunks), 1)
        self.assertIn("AI Engineer", chunks[0].text)
        self.assertEqual(chunks[0].metadata["source"], "Uploaded file: sample-jd.txt")
        self.assertEqual(chunks[0].metadata["file_name"], "sample-jd.txt")
        self.assertTrue(chunks[0].metadata["uploaded"])
        self.assertIsNone(chunks[0].distance)

    def test_empty_upload_is_rejected(self) -> None:
        with self.assertRaisesRegex(UploadedDocumentError, "empty"):
            load_uploaded_file_chunks("jd.txt", b"", settings=_settings())

    def test_unsupported_upload_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(UploadedDocumentError, "Unsupported file type"):
            load_uploaded_file_chunks("jd.exe", b"hello", settings=_settings())


if __name__ == "__main__":
    unittest.main()
