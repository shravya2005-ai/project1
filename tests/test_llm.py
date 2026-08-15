import unittest
from src.llm import answer_question, _synthesize_local_grounded_answer


class TestLLM(unittest.TestCase):

    def test_answer_question_no_context(self):
        answer = answer_question("What is machine learning?", "", answer_only_from_docs=True)
        self.assertEqual(answer, "The information is not available in the uploaded documents.")

    def test_answer_question_grounded_synthesis(self):
        context = (
            "--- Source: deep_learning.pdf | Page: 4 ---\n"
            "Deep learning is a subset of machine learning based on artificial neural networks with representation learning."
        )
        sources = [{"source": "deep_learning.pdf", "page": 4, "similarity": 0.95, "snippet": context}]

        answer = answer_question(
            question="What is deep learning?",
            context=context,
            sources=sources,
            answer_only_from_docs=True,
            backend="local",
        )

        self.assertIn("deep_learning.pdf", answer)
        self.assertTrue("neural networks" in answer or "representation learning" in answer or "subset" in answer)


if __name__ == "__main__":
    unittest.main()
