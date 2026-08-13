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

LLM_BACKEND = os.getenv("LLM_BACKEND", "local").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "distilgpt2")
LOCAL_LLM_TASK = os.getenv("LOCAL_LLM_TASK", "text-generation")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
TOP_K = int(os.getenv("TOP_K", "4"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))

# Local persistence settings for Chroma
CHROMA_PERSIST_DIRECTORY = str(VECTOR_DIR)
