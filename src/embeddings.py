from functools import lru_cache
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

_model_instance = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy loads and caches the SentenceTransformer embedding model."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model_instance


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generates normalized vector embeddings for a list of text strings."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    if isinstance(embeddings, np.ndarray):
        return embeddings.tolist()
    return embeddings

