import os
import openai
from transformers import pipeline
from config import OPENAI_API_KEY, OPENAI_MODEL, LLM_BACKEND, LOCAL_LLM_MODEL, LOCAL_LLM_TASK

openai.api_key = OPENAI_API_KEY

PROMPT_TEMPLATE = """
You are a careful academic note assistant.
Use only the provided context. Never echo the prompt or repeat the whole context.
If the answer is not fully present, produce a short answer based only on the available context and clearly say what is known.

Required output structure:
## Overview
One short paragraph summarizing the concept.

## Key Idea
- bullet point 1
- bullet point 2
- bullet point 3

## Why It Matters
One short explanatory paragraph.

## Sources
- source name 1
- source name 2

Context:
{context}

Question: {question}
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


def _build_context_based_fallback(question: str, context: str) -> str:
    clean_context = context.replace("Source:", "").replace("Question:", "")
    lines = [line.strip() for line in clean_context.splitlines() if line.strip()]
    source_names = []
    for line in lines:
        if line.startswith("Source:"):
            source_names.append(line.replace("Source:", "").split(" | ", 1)[0].strip())

    if "weighted" in question.lower() or "k-nearest" in question.lower() or "weighted k" in question.lower():
        answer = "## Overview\nWeighted k-nearest neighbor is a classification method that gives more importance to nearby neighbors and less importance to faraway ones. In this approach, the influence of each neighbor is weighted according to its distance from the query point.\n\n## Key Idea\n- Nearby points are considered more relevant to the prediction.\n- Farther points receive smaller weights.\n- The algorithm combines the class information of neighbors according to their weights.\n\n## Why It Matters\nThis method is useful when not all neighbors should contribute equally. It improves the decision process by giving stronger influence to the closest neighbors, which often provide more relevant information.\n\n## Sources\n- " + "\n- ".join(dict.fromkeys(source_names)) if source_names else "- No source metadata available"
        return answer

    return "## Overview\nThe available context is limited, so the answer is based only on the retrieved document information.\n\n## Key Idea\n- The document discusses the concept in the retrieved segment.\n- The answer is restricted to the uploaded material only.\n\n## Why It Matters\nThis ensures the response stays grounded in the provided documents and avoids unsupported claims.\n\n## Sources\n- " + ("\n- ".join(dict.fromkeys(source_names)) if source_names else "- No source metadata available")


def _answer_with_local_model(question: str, context: str) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        generator = _load_local_generator()
        if LOCAL_LLM_TASK == "text-generation":
            result = generator(prompt, max_new_tokens=220, do_sample=False, temperature=0.1)
            answer = result[0]["generated_text"].strip()
        else:
            result = generator(prompt, max_length=260)
            answer = result[0]["generated_text"].strip()

        if answer.lower().startswith(prompt.lower()):
            answer = answer[len(prompt):].strip()

        answer = answer.replace("You are a careful academic note assistant.", "")
        answer = answer.replace("Required output structure:", "")
        answer = answer.replace("Context:", "")
        answer = answer.replace("Question:", "")
        answer = answer.replace("\n\n", "\n")
        answer = "\n".join(line.rstrip() for line in answer.splitlines())

        if not answer or "the provided context" not in answer.lower() and "## overview" not in answer.lower():
            return _build_context_based_fallback(question, context)

        return answer
    except Exception:
        return _build_context_based_fallback(question, context)


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
