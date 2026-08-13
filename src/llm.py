import os
import openai
from transformers import pipeline
from config import OPENAI_API_KEY, OPENAI_MODEL, LLM_BACKEND, LOCAL_LLM_MODEL, LOCAL_LLM_TASK

openai.api_key = OPENAI_API_KEY

PROMPT_TEMPLATE = """
You are a document-based answer assistant.
Answer only using the provided context. Do not repeat the context or the prompt.
If the answer is not present in the context, say: "The information is not available in the uploaded documents."

Context:
{context}

Question: {question}

Answer in 3 to 5 clear sentences, using the exact information from the context.
"""

_local_generator = None


def _load_local_generator():
    global _local_generator
    if _local_generator is None:
        _local_generator = pipeline(
            LOCAL_LLM_TASK,
            model=LOCAL_LLM_MODEL,
            device=-1,
        )
    return _local_generator


def _answer_with_local_model(question: str, context: str) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        generator = _load_local_generator()
        if LOCAL_LLM_TASK == "text-generation":
            result = generator(prompt, max_new_tokens=120, do_sample=False, temperature=0.1)
            answer = result[0]["generated_text"].strip()
        else:
            result = generator(prompt, max_length=200)
            answer = result[0]["generated_text"].strip()

        if answer.lower().startswith(prompt.lower()):
            answer = answer[len(prompt):].strip()

        answer = answer.replace("Context:", "").replace("Question:", "").replace("Answer in 3 to 5 clear sentences, using the exact information from the context.", "").strip()
        answer = answer.replace("\n\n", "\n")

        if not answer:
            return "No answer could be generated from the local model."
        return answer
    except Exception as exc:
        return f"Local model error: {exc}"


def answer_question(question: str, context: str, answer_only_from_docs: bool = True, sources=None) -> str:
    if answer_only_from_docs and not context.strip():
        return "No relevant information was found in the uploaded documents."

    if LLM_BACKEND == "openai" and OPENAI_API_KEY:
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
            message = str(exc).lower()
            if "credit" in message or "billing" in message or "insufficient_quota" in message:
                return "OpenAI account has no remaining credits. Add billing credits or use a different API key with available quota."
            if "api key" in message or "authentication" in message or "invalid key" in message:
                return "OpenAI API key is invalid or expired. Update the OPENAI_API_KEY in your .env file."
            if "model" in message and "not found" in message:
                return f"The model '{OPENAI_MODEL}' is not available for your account. Update OPENAI_MODEL in .env."
            return f"Error generating answer: {exc}"

    if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
        return "OpenAI API key is missing. Set OPENAI_API_KEY in .env or change LLM_BACKEND=local."

    return _answer_with_local_model(question, context)
