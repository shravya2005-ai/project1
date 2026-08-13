import os
import re
import streamlit as st
from pathlib import Path
from datetime import datetime

from config import UPLOAD_DIR, OPENAI_API_KEY, LLM_BACKEND
from src.chat_history import ChatHistory
from src.document_processor import extract_texts_from_file, extract_pdf_diagrams, SUPPORTED_EXTENSIONS
from src.vector_store import VectorStore
from src.llm import answer_question

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide",
)

def sanitize_markdown_answer(text: str) -> str:
    if not text:
        return text

    text = re.sub(r"\[\]\(https?://localhost:\d+/#?[A-Za-z0-9\-_]+\)", "", text)
    text = re.sub(r"\[\]\(#[A-Za-z0-9\-_]+\)", "", text)
    text = re.sub(r"\s+\[\]\(.*?\)", "", text)
    return text.strip()


def markdown_to_html(text: str) -> str:
    text = sanitize_markdown_answer(text)
    if not text:
        return ""

    lines = text.splitlines()
    html_lines = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            continue

        if stripped.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
            continue

        if stripped.startswith("### "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<h3>{stripped[4:]}</h3>")
            continue

        if stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{stripped[2:]}</li>")
            continue

        if in_list:
            html_lines.append("</ul>")
            in_list = False

        html_lines.append(f"<p>{stripped}</p>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f5f8ff 0%, #edf4ff 35%, #f9fbff 100%); }
    div[data-testid="stSidebar"] { background: rgba(255,255,255,0.9); border-right: 1px solid rgba(103, 116, 148, 0.15); }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .hero {
        background: linear-gradient(135deg, rgba(72, 118, 255, 0.14), rgba(139, 92, 246, 0.08));
        border: 1px solid rgba(72, 118, 255, 0.12);
        border-radius: 20px;
        padding: 2rem 2rem 1.5rem 2rem;
        box-shadow: 0 18px 40px rgba(31, 66, 135, 0.08);
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: #172033; font-size: 2.5rem; font-weight: 800; margin-bottom: 0.4rem; }
    .hero p { color: #46546d; font-size: 1rem; }
    .status-card {
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(90, 107, 145, 0.12);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 6px 18px rgba(16, 24, 40, 0.04);
    }
    .source-box {
        background: #f8faff;
        border: 1px solid rgba(90, 107, 145, 0.12);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-top: 0.6rem;
        color: #1d2a3d;
    }
    div[data-testid="stMarkdownContainer"] {
        background: #ffffff;
        border: 1px solid rgba(86, 120, 255, 0.18);
        border-radius: 18px;
        padding: 1.2rem 1.1rem;
        margin-top: 1rem;
        color: #1a2438;
        box-shadow: 0 10px 24px rgba(31, 66, 135, 0.05);
    }
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4 {
        color: #1a2438;
        margin-top: 0.3rem;
        margin-bottom: 0.6rem;
    }
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {
        color: #2d3b52;
        font-size: 0.98rem;
        line-height: 1.7;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #6d5ef5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.1rem;
        box-shadow: 0 8px 18px rgba(59, 130, 246, 0.2);
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.9);
        color: #1b2433;
        border: 1px solid rgba(104, 120, 150, 0.25);
        border-radius: 10px;
    }
    .stFileUploader > div { background: rgba(255,255,255,0.7); border-radius: 12px; }
    .stCheckbox { color: #22314d; }
    </style>
    """,
    unsafe_allow_html=True,
)

if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
    st.sidebar.error("Set OPENAI_API_KEY in .env before using the app, or switch to LLM_BACKEND=local.")
else:
    st.sidebar.info(f"LLM backend: {LLM_BACKEND.upper()}")

storage = VectorStore()
history = ChatHistory()

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = history.list_documents()

st.sidebar.title("Document Assistant")
st.sidebar.markdown("Upload PDF, DOCX or TXT files and ask questions against them.")

with st.sidebar.expander("Upload new documents"):
    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=list(SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
    )
    if st.button("Upload files") and uploaded_files:
        for uploaded in uploaded_files:
            file_path = UPLOAD_DIR / uploaded.name
            if file_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                file_path = UPLOAD_DIR / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            with open(file_path, "wb") as f:
                f.write(uploaded.getbuffer())
            chunks = extract_texts_from_file(file_path)
            if not chunks:
                st.warning(f"No text could be extracted from {uploaded.name}.")
                continue
            storage.add_chunks(chunks)
            history.save_document(uploaded.name, str(file_path), datetime.now().isoformat())
            st.success(f"Uploaded and indexed {uploaded.name}")
        st.session_state.uploaded_files = history.list_documents()

st.sidebar.markdown("---")
st.sidebar.subheader("Uploaded documents")
for doc in st.session_state.uploaded_files:
    col1, col2 = st.sidebar.columns([4, 1])
    col1.write(doc[0])
    if col2.button("Delete", key=f"delete_{doc[0]}"):
        history.delete_document(doc[0])
        storage.delete_by_source(doc[0])
        st.session_state.uploaded_files = history.list_documents()
        st.success(f"Removed {doc[0]}")

st.sidebar.markdown("---")
answer_only = st.sidebar.checkbox("Answer only from uploaded documents", value=True)

st.markdown(
    """
    <div class="hero">
        <h1>RAG Document Assistant</h1>
        <p>Upload notes, PDFs, and study materials, then ask natural-language questions to get grounded answers from your documents.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

question = st.text_input("Ask a question", key="question_input")
answer_length = st.radio(
    "Answer style",
    options=[
        ("Short", "Quick revision"),
        ("Medium", "Balanced summary"),
        ("Detailed", "Full explanation"),
    ],
    index=1,
    horizontal=True,
    format_func=lambda x: x[0] if isinstance(x, tuple) else x,
)
answer_length = answer_length[0] if isinstance(answer_length, tuple) else answer_length
if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.uploaded_files:
        st.warning("Upload at least one document before asking a question.")
    else:
        results = storage.search(question)
        if not results:
            st.info("No relevant content was found in the uploaded documents.")
        else:
            context = []
            sources = []
            for item in results:
                metadata = item["metadata"]
                source_text = (
                    f"Source: {metadata.get('source')}"
                    + (f" | Page: {metadata.get('page')}" if metadata.get("page") else "")
                    + f"\n{item['document']}"
                )
                context.append(source_text)
                sources.append(metadata.get("source"))
            joined_context = "\n\n".join(context)
            answer = answer_question(
                question,
                joined_context,
                answer_only_from_docs=answer_only,
                sources=list(dict.fromkeys(sources)),
                answer_length=answer_length,
            )
            st.markdown("### Answer")
            st.markdown(markdown_to_html(answer), unsafe_allow_html=True)

            source_map = {name: path for name, path in history.list_documents()}
            diagram_items = []
            for source_name in list(dict.fromkeys(sources)):
                source_path = source_map.get(source_name)
                if not source_path:
                    continue
                try:
                    docs = extract_pdf_diagrams(Path(source_path), max_images=3)
                    diagram_items.extend(docs)
                except Exception:
                    continue

            if diagram_items:
                st.markdown("### Diagrams")
                for diagram in diagram_items:
                    st.image(
                        diagram["image_bytes"],
                        caption=f"{diagram['source']} — page {diagram['page']}",
                        use_container_width=True,
                    )

            st.markdown("### Source excerpts")
            for item in results:
                metadata = item["metadata"]
                st.markdown(
                    f"<div class='source-box'><strong>{metadata.get('source')}</strong>"
                    + (f" – page {metadata.get('page')}" if metadata.get("page") else "")
                    + f"<br>{item['document']}</div>",
                    unsafe_allow_html=True,
                )
