import uuid
from typing import Dict, List, Optional
import chromadb
from config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY, TOP_K
from src.embeddings import embed_texts


class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIRECTORY, collection_name: str = CHROMA_COLLECTION_NAME):
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, chunks: List[Dict]) -> int:
        """Embeds and indexes document chunks in ChromaDB."""
        if not chunks:
            return 0

        texts = [item["document"] for item in chunks]
        metadatas = [item["metadata"] for item in chunks]

        # Generate unique IDs for each chunk to avoid collisions
        ids = []
        for i, meta in enumerate(metadatas):
            if "chunk_id" in meta:
                ids.append(str(meta["chunk_id"]))
            else:
                source = meta.get("source", "doc")
                page = meta.get("page", 1)
                ids.append(f"{source}_p{page}_c{i}_{uuid.uuid4().hex[:6]}")

        embeddings = embed_texts(texts)

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )
        return len(texts)

    def search(self, query: str, k: int = TOP_K) -> List[Dict]:
        """Performs semantic similarity vector search for a given query string."""
        if not query.strip():
            return []

        total_docs = self.collection.count()
        if total_docs == 0:
            return []

        k = min(k, total_docs)
        query_embeddings = embed_texts([query])
        if not query_embeddings:
            return []

        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        hits = []
        for document, metadata, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )
        return hits

    def delete_by_source(self, source_name: str) -> bool:
        """Deletes all vector store entries belonging to a given source document."""
        try:
            self.collection.delete(where={"source": source_name})
            return True
        except Exception as exc:
            print(f"Error deleting vectors for {source_name}: {exc}")
            return False

    def get_count(self) -> int:
        """Returns total number of chunks indexed in the collection."""
        return self.collection.count()

    def clear_all(self):
        """Clears all document vectors in the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(self.collection_name)
        except Exception:
            pass

