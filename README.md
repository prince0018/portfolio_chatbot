# Portfolio Chatbot

This project uses a persistent ChromaDB vector index for personal portfolio data.

## Setup

```bash
source .venv/bin/activate
python -m pip install -e .
```

## Create the ChromaDB index

```bash
python scripts/create_chroma_index.py
```

By default, the index is stored in `./chroma_db` and the collection is named
`portfolio_profile`.

You can override these values with environment variables:

```bash
CHROMA_DB_PATH=./chroma_db CHROMA_COLLECTION_NAME=portfolio_profile python scripts/create_chroma_index.py
```

## Add Source Documents

Put your source files in `docs/`.

Supported file types:

- `.pdf`
- `.docx`
- `.txt`
- `.md`

Examples:

```text
docs/resume.pdf
docs/profile.docx
docs/projects.md
```

## Configure OpenAI Embeddings

Copy `.env.example` to a local `.env` file, then add your real OpenAI API key there.

`python-dotenv` loads `.env`. The `.env.example` file is only a template and should not contain
your real secret key.

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_CHAT_MODEL=gpt-4o-mini
```

The default embedding model is `text-embedding-ada-002`.

## Ingest Documents

```bash
python scripts/ingest_documents.py
```

The ingestion pipeline:

1. Reads files from `docs/`.
2. Extracts text from PDF, Word, text, and Markdown files.
3. Splits text with LangChain `RecursiveCharacterTextSplitter`.
4. Creates OpenAI embeddings for each chunk.
5. Stores chunks, embeddings, and metadata in ChromaDB.

Default chunk settings:

```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
OPENAI_EMBEDDING_BATCH_SIZE=64
```

## Chat With Your Documents

After ingestion, start the chatbot:

```bash
python scripts/chat.py
```

Then ask questions in the terminal:

```text
You: What are Prince's main skills?
```

Type `exit` to stop.

You can also ask one question directly:

```bash
python scripts/chat.py "What projects has Prince worked on?"
```

The chatbot uses semantic search:

1. Embeds your question with the same embedding model used during ingestion.
2. Retrieves the most similar chunks from ChromaDB.
3. Sends only those chunks to the OpenAI chat model.
4. Answers based on the retrieved document context.

Retrieval settings:

```env
RETRIEVAL_TOP_K=4
MAX_HISTORY_TURNS=4
```

## Run the Web Frontend

Start the local web app:

```bash
python scripts/serve.py
```

Open:

```text
http://127.0.0.1:8000
```

The frontend is served from `frontend/` and calls:

```text
POST /api/chat
POST /api/chat/file
GET /api/status
```

The web chat also supports temporary file context for a single session. Attach a `.pdf`,
`.docx`, `.txt`, or `.md` file such as a job description, then ask questions like:

```text
Does this JD suit Prince's qualifications?
Give me interview questions based on this JD.
```

Uploaded files are parsed for the current chat request and are not written into the
persistent ChromaDB portfolio index.

## Run Tests

```bash
python -m compileall src scripts
python -m unittest discover -s tests -v
```

The GitHub Actions workflow in `.github/workflows/ci.yml` runs these checks on every
push and pull request.
.