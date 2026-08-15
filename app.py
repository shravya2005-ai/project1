import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import (
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    GEMINI_API_KEY,
    LLM_BACKEND,
    OPENAI_API_KEY,
    RELEVANCE_THRESHOLD,
    TOP_K,
    UPLOAD_DIR,
)
import importlib
import src.llm
importlib.reload(src.llm)
from src.llm import answer_question
from src.chat_history import ChatHistory
from src.document_processor import SUPPORTED_EXTENSIONS, extract_texts_from_file
from src.retriever import RAGRetriever
from src.vector_store import VectorStore


# Set Streamlit Page Configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Light Theme Styling (CSS)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main app light theme background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
        color: #0f172a;
    }
    
    /* Clean Light Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.02);
    }
    
    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-top: 4px solid #4f46e5;
        border-radius: 16px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        color: #1e1b4b;
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 50%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .hero-subtitle {
        color: #475569;
        font-size: 1.02rem;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
    
    /* Stat Badge Chips */
    .badge-bar {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    
    .stat-chip {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 30px;
        padding: 0.35rem 0.9rem;
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .stat-chip-active {
        background: #e0e7ff;
        border-color: #c7d2fe;
        color: #3730a3;
    }
    
    /* Quick Prompt Chips */
    .prompt-chip {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-size: 0.88rem;
        color: #1e293b;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    
    /* Citation Cards */
    .citation-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #4f46e5;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-top: 0.6rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    
    .citation-meta {
        font-size: 0.84rem;
        font-weight: 700;
        color: #4f46e5;
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.4rem;
    }
    
    .citation-snippet {
        font-size: 0.88rem;
        color: #334155;
        font-style: italic;
        line-height: 1.5;
    }
    
    /* Progress bar style in Streamlit */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
    }
    
    /* Tabs customization */
    button[data-baseweb="tab"] {
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        color: #64748b !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
    }
    
    button[aria-selected="true"] {
        color: #4f46e5 !important;
        background: #e0e7ff !important;
        border-bottom: 2px solid #4f46e5 !important;
    }
    
    /* Input Field Styling */
    div[data-baseweb="input"] {
        border-radius: 12px !important;
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important;
    }
    
    /* Custom buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Initialize Core Services
vector_store = VectorStore()
retriever = RAGRetriever(vector_store=vector_store)
chat_history = ChatHistory()

# Streamlit Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = chat_history.get_messages()

if "documents" not in st.session_state:
    st.session_state.documents = chat_history.list_documents()

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# SIDEBAR: File Upload & Management Workspace
st.sidebar.markdown("<h2 style='font-size: 1.3rem; font-weight: 800; color: #0f172a; margin-bottom: 0.2rem;'>📁 Knowledge Workspace</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 0.85rem; color: #475569; margin-bottom: 1rem;'>Upload PDF, DOCX, or TXT documents to build vector index.</p>", unsafe_allow_html=True)

with st.sidebar.expander("⬆️ Upload & Index Documents", expanded=True):
    uploaded_files = st.file_uploader(
        "Supported formats: PDF, DOCX, TXT",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        key="file_uploader",
    )
    if st.button("🚀 Process & Index Files", use_container_width=True) and uploaded_files:
        with st.spinner("Extracting text & generating vector embeddings..."):
            added_count = 0
            for uploaded in uploaded_files:
                file_path = UPLOAD_DIR / uploaded.name
                if file_path.exists():
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    file_path = UPLOAD_DIR / f"{file_path.stem}_{timestamp}{file_path.suffix}"

                with open(file_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                chunks = extract_texts_from_file(file_path)
                if not chunks:
                    st.warning(f"No text extracted from {uploaded.name}")
                    continue

                indexed_count = vector_store.add_chunks(chunks)
                chat_history.save_document(
                    name=uploaded.name,
                    path=str(file_path),
                    size_bytes=len(uploaded.getbuffer()),
                    chunk_count=indexed_count,
                )
                added_count += 1

            st.session_state.documents = chat_history.list_documents()
            st.success(f"Indexed {added_count} document(s) successfully!")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-size: 1rem; font-weight: 700; color: #1e293b;'>📚 Active Documents</h3>", unsafe_allow_html=True)

docs = st.session_state.documents
if not docs:
    st.sidebar.info("No documents uploaded yet.")
else:
    for doc in docs:
        col1, col2 = st.sidebar.columns([4, 1])
        size_kb = doc["size_bytes"] / 1024.0 if doc["size_bytes"] else 0
        col1.markdown(f"📄 **{doc['name']}**  \n`<span style='color: #64748b; font-size: 0.78rem;'>{size_kb:.1f} KB • {doc['chunk_count']} chunks</span>`", unsafe_allow_html=True)
        if col2.button("🗑️", key=f"del_{doc['name']}", help=f"Delete {doc['name']}"):
            vector_store.delete_by_source(doc["name"])
            chat_history.delete_document(doc["name"])
            st.session_state.documents = chat_history.list_documents()
            st.sidebar.success(f"Removed {doc['name']}")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-size: 1rem; font-weight: 700; color: #1e293b;'>⚙️ Model & Controls</h3>", unsafe_allow_html=True)


# Backend Selection
backend_choice = st.sidebar.selectbox(
    "LLM Provider",
    options=["local", "openai", "gemini", "ollama"],
    index=["local", "openai", "gemini", "ollama"].index(LLM_BACKEND) if LLM_BACKEND in ["local", "openai", "gemini", "ollama"] else 0,
    help="Select LLM provider. 'local' works offline without API keys.",
)

user_api_key = None
if backend_choice == "openai":
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        user_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Paste your sk-... key here")
        if not user_api_key:
            st.sidebar.warning("⚠️ OPENAI_API_KEY not set. Using Local Grounded Mode.")
    else:
        user_api_key = OPENAI_API_KEY

elif backend_choice == "gemini":
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        user_api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Paste your AIza... key here")
        if not user_api_key:
            st.sidebar.warning("⚠️ GEMINI_API_KEY not set. Using Local Grounded Mode.")
    else:
        user_api_key = GEMINI_API_KEY


# Mode & Style Toggles
answer_only_docs = st.sidebar.checkbox(
    "Answer ONLY from documents",
    value=True,
    help="If enabled, refuses to answer questions not covered by the uploaded documents.",
)

answer_length_choice = st.sidebar.radio(
    "Response Length / Detail",
    options=["Short", "Medium", "Detailed"],
    index=1,
    horizontal=True,
)

st.sidebar.markdown("---")
if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    chat_history.clear_messages()
    st.session_state.messages = []
    st.rerun()

# HERO BANNER & STATS
total_vectors = vector_store.get_count()
total_docs_count = len(docs)

st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-title">⚡ RAG Document Assistant</div>
        <div class="hero-subtitle">Upload PDFs, DOCX, or TXT study materials to ask questions and receive context-grounded AI answers with exact source citations.</div>
        <div class="badge-bar">
            <div class="stat-chip stat-chip-active">📚 {total_docs_count} Indexed Documents</div>
            <div class="stat-chip">🧩 {total_vectors} Chunks in Vector Store</div>
            <div class="stat-chip">🔍 Embedding: {EMBEDDING_MODEL_NAME}</div>
            <div class="stat-chip">🤖 Provider: {backend_choice.upper()}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# TABBED WORKSPACE LAYOUT
tab_chat, tab_documents, tab_architecture = st.tabs(["💬 RAG Chat Assistant", "📊 Vector Index & Knowledge Base", "🛠️ System Architecture"])

with tab_chat:
    # Render Quick Starter Prompts
    st.markdown("<p style='font-size: 0.9rem; font-weight: 600; color: #94a3b8; margin-bottom: 0.5rem;'>Quick Starter Prompts:</p>", unsafe_allow_html=True)
    col_p1, col_p2, col_p3 = st.columns(3)
    if col_p1.button("📌 Summarize key concepts from documents", use_container_width=True):
        st.session_state.pending_prompt = "Summarize key concepts from documents"
        st.rerun()
    if col_p2.button("💡 Explain the main topics & definitions", use_container_width=True):
        st.session_state.pending_prompt = "Explain the main topics and definitions in the uploaded material"
        st.rerun()
    if col_p3.button("❓ What are the important points for study?", use_container_width=True):
        st.session_state.pending_prompt = "What are the most important study points and takeaways?"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📌 Source Citations & References", expanded=False):
                    for src in msg["sources"]:
                        sim_pct = int(src.get("similarity", 0) * 100)
                        page_str = f" (Page {src['page']})" if src.get("page") else ""
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <div class="citation-meta">
                                    <span>📄 {src['source']}{page_str}</span>
                                    <span>Relevance: {sim_pct}%</span>
                                </div>
                                <div class="citation-snippet">"{src['snippet']}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Handle query input (either from chat_input or pending quick prompt)
    user_query = st.chat_input("Ask a question about your uploaded documents...")
    if st.session_state.pending_prompt:
        user_query = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if user_query:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": user_query, "sources": None})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Process assistant answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document vectors & generating answer..."):
                docs_in_kb = chat_history.list_documents()

                if not docs_in_kb:
                    answer = "No documents found in knowledge base. Please upload at least one PDF/DOCX/TXT file in the sidebar first."
                    sources = []
                else:
                    hits = retriever.retrieve(
                        query=user_query,
                        top_k=TOP_K,
                        relevance_threshold=RELEVANCE_THRESHOLD,
                        filter_threshold=answer_only_docs,
                    )

                    if not hits:
                        answer = "The information is not available in the uploaded documents."
                        sources = []
                    else:
                        context_str, sources = retriever.format_context(hits)
                        answer = answer_question(
                            question=user_query,
                            context=context_str,
                            answer_only_from_docs=answer_only_docs,
                            sources=sources,
                            answer_length=answer_length_choice,
                            backend=backend_choice,
                            api_key=user_api_key,
                        )


                # Render assistant response
                st.markdown(answer)

                if sources:
                    with st.expander("📌 Source Citations & References", expanded=False):
                        for src in sources:
                            sim_pct = int(src.get("similarity", 0) * 100)
                            page_str = f" (Page {src['page']})" if src.get("page") else ""
                            st.markdown(
                                f"""
                                <div class="citation-card">
                                    <div class="citation-meta">
                                        <span>📄 {src['source']}{page_str}</span>
                                        <span>Relevance: {sim_pct}%</span>
                                    </div>
                                    <div class="citation-snippet">"{src['snippet']}"</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Save message pair to SQLite database history
                chat_history.save_message(role="user", content=user_query)
                chat_history.save_message(role="assistant", content=answer, sources=sources)
                st.session_state.messages = chat_history.get_messages()

with tab_documents:
    st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; color: #f1f5f9;'>📊 Knowledge Base Documents & Vectors</h3>", unsafe_allow_html=True)
    if not docs:
        st.info("No documents indexed yet. Use the sidebar to upload files.")
    else:
        for doc in docs:
            size_kb = doc["size_bytes"] / 1024.0 if doc["size_bytes"] else 0
            with st.expander(f"📄 {doc['name']} ({size_kb:.1f} KB — {doc['chunk_count']} chunks)", expanded=False):
                st.write(f"**File Path:** `{doc['path']}`")
                st.write(f"**Uploaded At:** `{doc['uploaded_at']}`")
                st.write(f"**Total Vector Chunks:** `{doc['chunk_count']}`")
                if st.button(f"Delete {doc['name']}", key=f"tab_del_{doc['name']}"):
                    vector_store.delete_by_source(doc["name"])
                    chat_history.delete_document(doc["name"])
                    st.session_state.documents = chat_history.list_documents()
                    st.success(f"Deleted {doc['name']}")
                    st.rerun()

with tab_architecture:
    st.markdown("<h3 style='font-size: 1.1rem; font-weight: 700; color: #f1f5f9;'>🛠️ RAG System Architecture</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        ```mermaid
        graph TD
            A[User Uploads PDF / DOCX / TXT] --> B[PyMuPDF / python-docx Text Extractor]
            B --> C[Recursive Paragraph & Sentence Chunker]
            C --> D[SentenceTransformer all-MiniLM-L6-v2 Embeddings]
            D --> E[(ChromaDB Vector Store)]
            
            F[User Natural-Language Question] --> G[Embed Question]
            G --> H[Semantic Similarity Search in ChromaDB]
            E --> H
            H --> I[RAGRetriever Filtering & Context Construction]
            I --> J[LLM Answer Synthesizer: OpenAI / Gemini / Ollama / Local]
            J --> K[Formatted Markdown Response + Source Citations]
        ```
        """,
        unsafe_allow_html=True,
    )
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        st.markdown("**Retriever Configuration:**")
        st.json({"TOP_K": TOP_K, "RELEVANCE_THRESHOLD": RELEVANCE_THRESHOLD, "COLLECTION": CHROMA_COLLECTION_NAME})
    with col_cfg2:
        st.markdown("**LLM Provider Configuration:**")
        st.json({"ACTIVE_BACKEND": backend_choice, "OPENAI_SET": bool(OPENAI_API_KEY), "GEMINI_SET": bool(GEMINI_API_KEY)})
