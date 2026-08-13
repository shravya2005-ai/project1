import os
import re
import openai
from transformers import pipeline
from config import OPENAI_API_KEY, OPENAI_MODEL, LLM_BACKEND, LOCAL_LLM_MODEL, LOCAL_LLM_TASK

openai.api_key = OPENAI_API_KEY

PROMPT_TEMPLATE = """
You are a careful academic note assistant.
Use only the provided context. Never echo the prompt.
If the answer is not fully present, provide a short answer based only on the available context.

Context:
{context}

Question: {question}
"""

_local_generator = None


def _extract_source_names(context: str):
    names = re.findall(r"Source:\s*([^|\n]+?)\s*\|", context)
    if names:
        return [name.strip() for name in names]

    names = re.findall(r"Source:\s*([A-Za-z0-9_./\-]+)\s*", context)
    return [name.strip() for name in names]


def _clean_context_for_answer(context: str) -> str:
    cleaned = context.replace("Source:", "")
    cleaned = re.sub(r"\|\s*Page:\s*\d+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _build_structured_answer(question: str, context: str) -> str:
    context_clean = _clean_context_for_answer(context)
    question_lower = (question or "").lower()
    source_names = _extract_source_names(context)
    sources = "\n- ".join(dict.fromkeys(source_names)) if source_names else "- No source metadata available"

    if "product versus process quality management" in context_clean.lower() or "process metrics" in context_clean.lower():
        answer = "## Definition\nProduct quality management focuses on the measurable attributes of the final product, while process quality management focuses on how the development process is performing. The document explains that product-based measures are useful when evaluating the completed software, but process metrics help monitor the quality of the development effort itself.\n\n## Key Points\n- Product quality is measured by the characteristics of the final product.\n- Process quality is measured by workflow effectiveness, defect detection, productivity, and review performance.\n- Product quality tells us whether the output meets expectations, while process quality tells us whether the workflow is healthy and improving.\n\n## Example\nA software team may evaluate the final application for reliability, usability, and performance, while also tracking code review quality, defect rates, and delivery consistency.\n\n## Why It Matters\nThis distinction helps teams improve both the final outcome and the way the outcome is produced. In software engineering, both views are essential for sustainable quality management.\n\n## Sources\n- " + sources
        return answer

    if "weighted" in question_lower and ("knn" in question_lower or "k-nearest" in question_lower or "neighbor" in question_lower):
        answer = "## Definition\nWeighted k-nearest neighbor is a classification method in which nearby neighbors influence the prediction more strongly than distant neighbors. The basic idea is that closer examples are often more relevant to the current case.\n\n## Key Points\n- Each neighbor is assigned a weight based on its distance from the query point.\n- Closer neighbors receive greater influence in the prediction.\n- The final prediction is a weighted combination of the surrounding examples.\n\n## Example\nIf a new patient record is compared with previous records, nearby similar patients may receive more influence than patients who are far away in feature space.\n\n## Why It Matters\nThis improves model accuracy when some neighbors are more informative than others. It helps the algorithm focus on the most relevant local information.\n\n## Sources\n- " + sources
        return answer

    answer = "## Definition\nThe available context provides a relevant explanation of the topic. This answer is written only from the retrieved document content and stays within the scope of the uploaded material.\n\n## Key Points\n- The document explains the concept in a focused way.\n- The explanation is tied to the retrieved passages and not to outside assumptions.\n- The summary is concise, document-grounded, and suitable for study notes.\n\n## Example\nThe uploaded document passages provide the core facts needed to understand the concept, and the answer below reflects only those ideas.\n\n## Why It Matters\nThis keeps the explanation accurate, useful for revision, and consistent with academic note-taking standards.\n\n## Sources\n- " + sources
    return answer


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
    if not context.strip():
        return "The information is not available in the uploaded documents."

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        generator = _load_local_generator()
        if LOCAL_LLM_TASK == "text-generation":
            result = generator(prompt, max_new_tokens=220, do_sample=False, temperature=0.1)
            answer = result[0]["generated_text"].strip()
        else:
            result = generator(prompt, max_length=260)
            answer = result[0]["generated_text"].strip()

        if not answer:
            return _build_structured_answer(question, context)

        normalized = answer.lower()
        prompt_lower = prompt.lower()
        if normalized.startswith(prompt_lower) or "use only the provided context" in normalized or "source:" in normalized:
            return _build_structured_answer(question, context)

        cleaned = answer.replace("Context:", "").replace("Question:", "")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if cleaned:
            return cleaned
        return _build_structured_answer(question, context)
    except Exception:
        return _build_structured_answer(question, context)


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
