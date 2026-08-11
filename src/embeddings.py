from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts):
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
