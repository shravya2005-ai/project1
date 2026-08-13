import chromadb
from chromadb.config import Settings
from config import CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME, TOP_K
from src.embeddings import embed_texts

class VectorStore:
    def __init__(self):
        settings = Settings(
            persist_directory=str(CHROMA_PERSIST_DIRECTORY),
            is_persistent=True,
        )
        self.client = chromadb.Client(settings=settings)
        self.collection = self._get_collection(CHROMA_COLLECTION_NAME)

    def _get_collection(self, name):
        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(name)

    def add_chunks(self, chunks):
        texts = [item["document"] for item in chunks]
        metadatas = [item["metadata"] for item in chunks]
        ids = [f"{metadata['source']}_{i}" for i, metadata in enumerate(metadatas)]
        embeddings = embed_texts(texts)
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

    def search(self, query, k=TOP_K):
        query_embedding = embed_texts([query])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results.get("documents"):
            return []

        hits = []
        for document, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            hits.append({
                "document": document,
                "metadata": metadata,
                "distance": distance,
            })
        return hits

    def delete_by_source(self, source_name):
        try:
            self.collection.delete(where={"source": source_name})
        except Exception:
            pass
