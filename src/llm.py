import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_BACKEND,
    LOCAL_LLM_MODEL,
    LOCAL_LLM_TASK,
    OLLAMA_ENDPOINT,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

SYSTEM_PROMPT = (
    "You are a helpful, precise RAG Document Assistant. "
    "Your primary goal is to answer the user's question accurately based strictly on the provided document context. "
    "If the answer is not present in the context and 'answer_only_from_docs' mode is enabled, state clearly that "
    "the information is not available in the uploaded documents. "
    "Cite source documents and page numbers where appropriate."
)

PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Length / Detail Level: {answer_length}

Instructions:
- Provide a clear, structured response matching the requested length ({answer_length}).
- Base your answer strictly on the context provided above.
- Use bullet points and clear markdown formatting for readability.
- If the question cannot be answered from the context, state: "The information is not available in the uploaded documents."
"""


def _clean_text(text: str) -> str:
    """Removes redundant whitespaces and formatting noise."""
    if not text:
        return ""
    cleaned = re.sub(r"\n{3,}", "\n\n", text)
    return cleaned.strip()


def _synthesize_local_grounded_answer(
    question: str,
    context: str,
    sources: Optional[List[Dict]] = None,
    answer_length: str = "Medium",
) -> str:
    """Dynamically synthesizes a grounded answer from retrieved context passages without fake/hardcoded strings."""
    if not context.strip():
        return "The information is not available in the uploaded documents."

    # Extract source names and page numbers from context headers or sources parameter
    source_refs = set()
    if sources:
        for s in sources:
            name = s.get("source", "Document")
            page = f" (Page {s['page']})" if s.get("page") else ""
            source_refs.add(f"{name}{page}")
    else:
        found_sources = re.findall(r"--- Source:\s*([^|\n-]+?)(?:\s*\|\s*Page:\s*(\d+))?\s*(?:\(.*?\))?\s*---", context)
        for name, page in found_sources:
            page_str = f" (Page {page})" if page else ""
            source_refs.add(f"{name.strip()}{page_str}")

    sources_str = "\n- ".join(sorted(source_refs)) if source_refs else "- Uploaded Document"

    # Split context into paragraphs/sentences
    raw_passages = [p.strip() for p in context.split("\n\n") if p.strip() and not p.startswith("--- Source:")]
    if not raw_passages:
        raw_passages = [context.strip()]

    # Score passages based on query keywords overlap
    q_words = set(re.findall(r"\w+", question.lower())) - {
        "what", "is", "are", "the", "a", "an", "in", "of", "and", "or", "to", "how", "why", "where", "which", "tell", "me", "about"
    }

    scored_passages = []
    for passage in raw_passages:
        p_words = set(re.findall(r"\w+", passage.lower()))
        overlap = len(q_words.intersection(p_words)) if q_words else 1
        scored_passages.append((overlap, passage))

    scored_passages.sort(key=lambda x: x[0], reverse=True)
    best_passages = [p for _, p in scored_passages[:4]]

    # Determine depth based on answer_length
    length = (answer_length or "Medium").lower()

    bullet_points = []
    for passage in best_passages:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", passage) if len(s.strip()) > 10]
        for sentence in sentences:
            if sentence not in bullet_points:
                bullet_points.append(sentence)
            if length == "short" and len(bullet_points) >= 2:
                break
            if length == "medium" and len(bullet_points) >= 4:
                break
            if length == "detailed" and len(bullet_points) >= 7:
                break

    if not bullet_points:
        bullet_points = [best_passages[0]] if best_passages else [context[:300]]

    bullets_markdown = "\n".join([f"- {bp}" for bp in bullet_points])

    if length == "short":
        return f"### Summary Answer\n{bullets_markdown}\n\n**Sources:**\n- {sources_str}"
    elif length == "detailed":
        return (
            f"### Overview\nBased on the uploaded document context, here is the detailed answer for: **{question}**\n\n"
            f"### Key Findings & Information\n{bullets_markdown}\n\n"
            f"### Source References\n- {sources_str}"
        )
    else:
        return (
            f"### Answer\n{bullets_markdown}\n\n"
            f"**Sources:**\n- {sources_str}"
        )


def _answer_with_openai(
    question: str,
    context: str,
    answer_length: str = "Medium",
    sources: Optional[List[Dict]] = None,
) -> str:
    """Generates an answer using the OpenAI API."""
    if not OPENAI_API_KEY:
        return "OpenAI API key is missing. Set OPENAI_API_KEY in your .env file or change LLM_BACKEND."

    prompt = PROMPT_TEMPLATE.format(context=context, question=question, answer_length=answer_length)

    try:
        import openai

        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            return _clean_text(response.choices[0].message.content)
        else:
            openai.api_key = OPENAI_API_KEY
            response = openai.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )
            return _clean_text(response.choices[0].message.content)

    except Exception as exc:
        err_msg = str(exc).lower()
        if "credit" in err_msg or "quota" in err_msg or "billing" in err_msg:
            return "OpenAI API error: Insufficient quota or billing issue. Falling back to local synthesizer."
        if "api_key" in err_msg or "authentication" in err_msg or "invalid" in err_msg:
            return "OpenAI API error: Invalid API Key. Please verify OPENAI_API_KEY in .env."
        return f"OpenAI generation error: {exc}"


def _answer_with_gemini(
    question: str,
    context: str,
    answer_length: str = "Medium",
) -> str:
    """Generates an answer using Google Gemini API."""
    if not GEMINI_API_KEY:
        return "Google Gemini API key is missing. Set GEMINI_API_KEY in your .env file."

    prompt = f"{SYSTEM_PROMPT}\n\n" + PROMPT_TEMPLATE.format(
        context=context, question=question, answer_length=answer_length
    )

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600},
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                text = candidates[0]["content"]["parts"][0]["text"]
                return _clean_text(text)
            return "No text returned from Gemini API."
    except Exception as exc:
        return f"Gemini API error: {exc}"


def _answer_with_ollama(
    question: str,
    context: str,
    answer_length: str = "Medium",
) -> str:
    """Generates an answer using a local Ollama instance."""
    prompt = f"{SYSTEM_PROMPT}\n\n" + PROMPT_TEMPLATE.format(
        context=context, question=question, answer_length=answer_length
    )

    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        req = urllib.request.Request(
            OLLAMA_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            return _clean_text(res_data.get("response", ""))
    except Exception as exc:
        return f"Ollama connection error ({OLLAMA_ENDPOINT}): {exc}"


def answer_question(
    question: str,
    context: str,
    answer_only_from_docs: bool = True,
    sources: Optional[List[Dict]] = None,
    answer_length: str = "Medium",
    backend: Optional[str] = None,
) -> str:
    """Main LLM answer generation entrypoint routing to specified backend with robust fallbacks."""
    if not question or not question.strip():
        return "Please enter a valid question."

    if answer_only_from_docs and not context.strip():
        return "The information is not available in the uploaded documents."

    active_backend = (backend or LLM_BACKEND).lower()

    if active_backend == "openai":
        res = _answer_with_openai(question, context, answer_length=answer_length, sources=sources)
        if not res.startswith("OpenAI API error"):
            return res
        # Fallback if OpenAI key fails
        return _synthesize_local_grounded_answer(question, context, sources=sources, answer_length=answer_length)

    elif active_backend == "gemini":
        res = _answer_with_gemini(question, context, answer_length=answer_length)
        if not res.startswith("Gemini API error"):
            return res
        return _synthesize_local_grounded_answer(question, context, sources=sources, answer_length=answer_length)

    elif active_backend == "ollama":
        res = _answer_with_ollama(question, context, answer_length=answer_length)
        if not res.startswith("Ollama connection error"):
            return res
        return _synthesize_local_grounded_answer(question, context, sources=sources, answer_length=answer_length)

    # Local backend / fallback
    return _synthesize_local_grounded_answer(question, context, sources=sources, answer_length=answer_length)
