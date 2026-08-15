import gc
import shutil
import unittest
from pathlib import Path
from config import VECTOR_DIR
from src.retriever import RAGRetriever
from src.vector_store import VectorStore


class TestRAGRetriever(unittest.TestCase):

    def test_rag_retriever(self):
        persist_dir = VECTOR_DIR / "test_retriever_vectorstore_db"
        if persist_dir.exists():
            shutil.rmtree(persist_dir, ignore_errors=True)

        try:
            vs = VectorStore(persist_dir=str(persist_dir), collection_name="test_retriever_coll")

            vs.add_chunks(
                [
                    {
                        "document": "Quantum computing utilizes qubits for complex computations.",
                        "metadata": {"source": "physics.pdf", "page": 3, "chunk_id": "physics_1"},
                    }
                ]
            )

            retriever = RAGRetriever(vector_store=vs)
            hits = retriever.retrieve("qubits quantum", top_k=1, relevance_threshold=0.1)

            self.assertEqual(len(hits), 1)
            self.assertGreater(hits[0]["similarity"], 0.0)

            context_str, sources = retriever.format_context(hits)
            self.assertIn("physics.pdf", context_str)
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]["source"], "physics.pdf")
        finally:
            del vs
            gc.collect()
            shutil.rmtree(persist_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
