import tempfile
import unittest
from pathlib import Path
from src.chat_history import ChatHistory


class TestChatHistory(unittest.TestCase):

    def test_chat_history_db_operations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ch = ChatHistory()
            ch.save_document(name="test_unit_doc.txt", path=str(Path(tmpdir) / "test_unit_doc.txt"), size_bytes=1024, chunk_count=5)

            docs = ch.list_documents()
            self.assertTrue(any(d["name"] == "test_unit_doc.txt" for d in docs))

            ch.save_message(role="user", content="Hello test prompt")
            ch.save_message(role="assistant", content="Hello test response", sources=[{"source": "test_unit_doc.txt"}])

            msgs = ch.get_messages()
            self.assertGreaterEqual(len(msgs), 2)
            self.assertEqual(msgs[-2]["role"], "user")
            self.assertEqual(msgs[-1]["role"], "assistant")
            self.assertEqual(msgs[-1]["sources"][0]["source"], "test_unit_doc.txt")

            ch.delete_document("test_unit_doc.txt")
            ch.clear_messages()
            self.assertEqual(len(ch.get_messages()), 0)


if __name__ == "__main__":
    unittest.main()
