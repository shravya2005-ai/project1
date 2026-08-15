from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vectorstore"
DB_PATH = DATA_DIR / "documents.db"
UPLOAD_DIR = DATA_DIR / "uploads"
ENV_PATH = BASE_DIR / ".env"

for path in (DATA_DIR, VECTOR_DIR, UPLOAD_DIR):
    path.mkdir(parents=True, exist_ok=True)

load_dotenv(ENV_PATH)

# LLM Backend options: 'openai', 'gemini', 'ollama', 'local'
LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()

# API Keys & Models
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "distilgpt2")
LOCAL_LLM_TASK = os.getenv("LOCAL_LLM_TASK", "text-generation")

# Embedding & Vector Database Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
CHROMA_PERSIST_DIRECTORY = str(VECTOR_DIR)

# Chunking & Retrieval Parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "4"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))

