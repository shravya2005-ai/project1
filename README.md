# RAG-Based Document Assistant 🤖

A modular, production-ready **Retrieval-Augmented Generation (RAG) Document Assistant** built with Python and Streamlit. Upload PDF, DOCX, or TXT documents, ask natural-language questions, and receive context-grounded answers complete with exact document and page citations.

---

## 🌟 Key Features

* **Multi-Format Document Ingestion**: Upload PDF (via `PyMuPDF`), Word (`python-docx`), and plain text (`TXT`) files.
* **Smart Text Chunking**: Recursive character and paragraph chunking with sentence boundary preservation and metadata attachment (`source`, `page`, `chunk_id`, `chunk_index`).
* **Persistent Vector Search**: Persistent **ChromaDB** client vector store powered by `SentenceTransformers` (`all-MiniLM-L6-v2`) embeddings.
* **Flexible LLM Engine**:
  * **OpenAI API** (`gpt-4o-mini`, `gpt-3.5-turbo`, `gpt-4o`)
  * **Google Gemini API** (`gemini-1.5-flash`)
  * **Ollama / Local API** (`llama3`)
  * **Local Grounded Synthesizer**: Zero-cost, zero-API-key offline fallback engine that synthesizes accurate, context-grounded markdown answers.
* **Strict "Answer Only from Documents" Mode**: Enforces relevance thresholds to prevent hallucinations and notifies users when content is not found in documents.
* **Interactive Streamlit Chat Interface**: Built using native `st.chat_message` and `st.chat_input` with persistent SQLite chat & document metadata history.
* **Source & Page References**: Collapsible expander citations detailing document name, page number, relevance percentage, and matching excerpt.
* **Complete Document Lifecycle Management**: Delete indexed documents cleanly from SQLite metadata, ChromaDB vector store, and physical disk storage.

---

## 🛠️ Project Structure

```text
rag-document-assistant/
├── app.py                      # Main Streamlit Web Application
├── config.py                   # Global configuration & environment settings
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment variables
├── README.md                   # Project documentation
├── data/                       # SQLite database & file upload directory
│   ├── documents.db            # Persistent SQLite database
│   └── uploads/                # Physical file storage
├── vectorstore/                # ChromaDB vector index persistence
├── src/                        # Modular application source code
│   ├── document_processor.py   # PDF/DOCX/TXT text extraction & recursive chunking
│   ├── embeddings.py           # SentenceTransformer model wrapper with lru_cache
│   ├── vector_store.py         # ChromaDB PersistentClient wrapper
│   ├── retriever.py            # RAG similarity retrieval & score calculation
│   ├── llm.py                  # Multi-backend LLM response synthesis & fallback
│   └── chat_history.py         # SQLite CRUD operations for documents and chat messages
└── tests/                      # Comprehensive unit test suite
    ├── test_document_processor.py
    ├── test_vector_store.py
    ├── test_retriever.py
    ├── test_llm.py
    └── test_chat_history.py
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites

- Python 3.9+
- Git

### 2. Installation

Clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/shravya2005-ai/project1.git
cd project1

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and set your desired parameters:

```bash
cp .env.example .env
```

Example `.env` configuration:

```env
# LLM Backend: 'local', 'openai', 'gemini', or 'ollama'
LLM_BACKEND=local

# Optional API Keys (Only needed if LLM_BACKEND is set to 'openai' or 'gemini')
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash

# Embeddings & Vector Database
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHROMA_COLLECTION_NAME=rag_documents

# Chunking & Retrieval Parameters
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TOP_K=4
RELEVANCE_THRESHOLD=0.35
```

### 4. Running the Web Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Running Unit Tests

Run the complete test suite:

```bash
python -m unittest discover tests
```

---

## 🏆 Resume Highlights

* **Architecture**: Fully modular, decoupled RAG pipeline design (`document_processor` -> `embeddings` -> `vector_store` -> `retriever` -> `llm` -> `chat_history`).
* **Vector Engineering**: Implemented deterministic chunk hashing to avoid vector collisions and cosine similarity scoring with thresholding.
* **Robustness & Fallbacks**: Multi-backend LLM router with automatic failover to local grounded synthesis when API quotas are exceeded.
* **Persistence & State Management**: Multi-tier storage architecture utilizing ChromaDB for vector embeddings, SQLite for chat history, and physical file handling.

---

## 📜 License

MIT License. Designed for academic research and portfolio demonstration.
