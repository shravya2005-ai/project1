# RAG-Based Document Assistant

A web-based Retrieval-Augmented Generation (RAG) Document Assistant built with Python. Upload PDF/DOCX/TXT documents, ask natural-language questions about their contents, and get answers that cite the source document and page when available.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment variables](#environment-variables)
  - [Run the app](#run-the-app)
- [Configuration & components](#configuration--components)
- [Development](#development)
- [Testing](#testing)
- [Deployment notes](#deployment-notes)
- [Contributing](#contributing)
- [License & Contact](#license--contact)

---

## Overview

This application enables users to upload one or more documents (PDF/DOCX/TXT), automatically extract and chunk the text, create semantic embeddings for chunks, store them in a vector store (ChromaDB), and answer user questions by retrieving the most relevant chunks and querying an LLM. The app returns the generated answer together with source references (document name and page/position when available).

## Key Features

- Upload multiple PDF / DOCX / TXT files
- Automatic text extraction (PyMuPDF for PDFs, python-docx for DOCX)
- Chunking of extracted text for better retrieval
- Embeddings using Sentence-Transformers
- Vector store with ChromaDB (semantic search)
- RAG-based question answering using an LLM (OpenAI or compatible)
- Chat-style interface with conversation history
- Option: "Answer only from uploaded documents" (no hallucinations)
- Source and page references for each answer
- Document management (upload, list, delete)
- Simple, clean Streamlit UI

## How it works

1. User uploads documents.
2. The app extracts text and metadata (filename, page numbers).
3. Text is split into chunks (with overlap).
4. Embeddings are generated per chunk via a sentence-transformer.
5. Embeddings + metadata are stored in ChromaDB.
6. On a user query: embed the query, run semantic search in ChromaDB, retrieve top-k chunks.
7. Pass retrieved context + user question to the LLM with a prompt that requests source citations.
8. Display the LLM answer and the source chunks (document name, page).

## Tech stack

- Language: Python 3.8+
- Frontend/UI: Streamlit
- Document processing: PyMuPDF (fitz), python-docx
- Embeddings: sentence-transformers (eg. all-MiniLM-L6-v2)
- Vector DB: ChromaDB
- LLM: OpenAI API (GPT-4/3.5) or any LLM accessible via API (optionally via LangChain)
- Persistence for metadata: SQLite
- Optional: LangChain for chaining retriever + LLM steps

## Project structure

Suggested structure (adapt as needed):

rag-document-assistant/
├── app.py                      # Streamlit app entrypoint
├── requirements.txt
├── README.md
├── config.py                   # central configuration
├── data/                       # uploaded raw files
├── vectorstore/                # chroma persistence directory
├── src/
│   ├── document_processor.py   # extract text and metadata
│   ├── embeddings.py           # generate embeddings
│   ├── vector_store.py         # ChromaDB wrapper
│   ├── retriever.py            # search + rerank logic
│   ├── llm.py                  # LLM prompt & call wrapper
│   └── chat_history.py         # persist/retrieve chat history (SQLite)
└── tests/

## Quick start

### Prerequisites

- Python 3.8+
- Git
- (Optional) OpenAI account + API key if using OpenAI LLM
- (Optional) chromadb dependencies (see requirements.txt)

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/shravya2005-ai/project1.git
   cd project1
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux / macOS
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Environment variables

Create a `.env` file or export these environment variables:

- OPENAI_API_KEY=sk-...
- CHROMA_PERSIST_DIR=./vectorstore
- EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
- DB_PATH=./data/chat_history.db

(If you use an alternative LLM provider or model, set the appropriate keys/URLs.)

Provide a sample `.env.example` file in the repo documenting required variables.

### Run the app (development)

Run the Streamlit app locally:
```bash
streamlit run app.py
```

Open the local Streamlit URL (usually http://localhost:8501).

## Configuration & components

- document_processor.py
  - Extracts text and metadata from PDFs (pages) and DOCX/TXT files.
  - Returns chunks with: text, doc_name, page_number, chunk_index.

- embeddings.py
  - Loads a sentence-transformers model and generates embeddings for chunks and queries.

- vector_store.py
  - Wrapper around ChromaDB for upsert, semantic search, persistence, and metadata retrieval.

- retriever.py
  - Performs top-k retrieval; optionally implements hybrid ranking or re-ranking.

- llm.py
  - Formats prompts, enforces "answer from documents only" when requested, calls LLM API and parses responses.

- chat_history.py
  - Stores conversation turns and references to retrieved sources in SQLite.

Important implementation notes:
- Chunk size: ~500 tokens with ~50-100 token overlap (tune experimentally).
- Persist Chroma with a directory so embeddings survive restarts.
- Include defenses in prompts to avoid hallucinations and instruct model to show source text & page.

## Development

- Linting / formatting:
  - Use black / isort / flake8 as your style tools.
- Local development tips:
  - Provide small test PDFs or TXT files under `tests/fixtures`.
  - Add unit tests for text extraction, chunking, embeddings integration, and retriever.

Example commands:
```bash
# run tests
pytest

# format
black .

# run dev server
streamlit run app.py
```

## Testing

- Unit tests for:
  - document_processor: correct extraction and page mapping
  - embeddings: embedding shapes and deterministic behavior
  - vector_store: upsert and search returns expected results
  - llm: prompt formatting (mock the LLM in tests)
- Integration tests:
  - Upload a small PDF and query for text known to exist; assert retrieved source matches.

## Deployment notes

- For production, consider:
  - Deploy Streamlit behind a web server or use Streamlit Cloud / Docker.
  - Use a managed/vector DB or persistent Chroma with backups.
  - Secure API keys (do not bake into images); use environment variables or secrets manager.
  - Rate-limit LLM calls and add caching for repeated queries.
  - Add authentication if allowing private/secure documents.

## Security & Privacy

- Uploaded documents may contain sensitive data — ensure storage is secure and configure retention/auto-delete policies.
- Ensure LLM provider privacy policy is acceptable for your data.
- Consider local-only LLMs or on-prem deployments for sensitive documents.

## Contributing

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature"
   git push origin feature/my-feature
   ```
4. Open a Pull Request and describe the change.

Add/maintain a `CONTRIBUTING.md` for more detailed contribution guidance.

## License & Contact

- License: MIT (create a LICENSE file if not present)
- Maintainer: shravya2005-ai
- Repo: https://github.com/shravya2005-ai/project1

---

If you'd like, I can:
- Commit this README to the repository for you.
- Generate a `requirements.txt` with pinned versions.
- Create starter implementations for `document_processor.py`, `vector_store.py`, and `app.py`.
- Add example tests and `.env.example`.
