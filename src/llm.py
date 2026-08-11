import os
import openai
from config import OPENAI_API_KEY, OPENAI_MODEL

openai.api_key = OPENAI_API_KEY

PROMPT_TEMPLATE = """
Use the provided context to answer the question. If the answer is not present in the context, say that the information cannot be found in the uploaded documents.

Context:
{context}

Question: {question}

Answer concisely and only based on the context.
"""


def answer_question(question: str, context: str, answer_only_from_docs: bool = True, sources=None) -> str:
    if answer_only_from_docs and not context.strip():
        return "No relevant information was found in the uploaded documents."

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    if sources:
        prompt += f"\n\nSources included: {', '.join(sources)}"

    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers only from user-provided documents."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"Error generating answer: {exc}"
