import gc
import shutil
import unittest
from pathlib import Path
from config import VECTOR_DIR
from src.vector_store import VectorStore


class TestVectorStore(unittest.TestCase):

    def test_vector_store_add_and_search(self):
        persist_dir = VECTOR_DIR / "test_vectorstore_db"
        if persist_dir.exists():
            shutil.rmtree(persist_dir, ignore_errors=True)

        try:
            vs = VectorStore(persist_dir=str(persist_dir), collection_name="test_collection")

            test_chunks = [
                {
                    "document": "Artificial Intelligence is transforming modern technology.",
                    "metadata": {"source": "ai_notes.txt", "page": 1, "chunk_id": "ai_notes_1"},
                },
                {
                    "document": "Photosynthesis is the process by which plants turn light into energy.",
                    "metadata": {"source": "bio_notes.txt", "page": 1, "chunk_id": "bio_notes_1"},
                },
            ]

            added = vs.add_chunks(test_chunks)
            self.assertEqual(added, 2)
            self.assertEqual(vs.get_count(), 2)

            results = vs.search("What is Artificial Intelligence?", k=2)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]["metadata"]["source"], "ai_notes.txt")

            vs.delete_by_source("bio_notes.txt")
            self.assertEqual(vs.get_count(), 1)
        finally:
            del vs
            gc.collect()
            shutil.rmtree(persist_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
