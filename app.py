import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from config import (
    GEMINI_API_KEY,
    LLM_BACKEND,
    OPENAI_API_KEY,
    RELEVANCE_THRESHOLD,
    TOP_K,
    UPLOAD_DIR,
)
from src.chat_history import ChatHistory
from src.document_processor import SUPPORTED_EXTENSIONS, extract_texts_from_file
from src.llm import answer_question
from src.retriever import RAGRetriever
from src.vector_store import VectorStore

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (CSS)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(147, 51, 234, 0.15) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }
    
    .doc-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
    }
    
    .source-tag {
        background: rgba(59, 130, 246, 0.2);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
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

# SIDEBAR: File Upload & Management
st.sidebar.title("📄 Document Workspace")
st.sidebar.markdown("Upload files to build your intelligent knowledge base.")

with st.sidebar.expander("⬆️ Upload New Documents", expanded=True):
    uploaded_files = st.file_uploader(
        "Supported: PDF, DOCX, TXT",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        key="file_uploader",
    )
    if st.button("Process & Index Documents", use_container_width=True) and uploaded_files:
        with st.spinner("Extracting text & generating vector embeddings..."):
            added_count = 0
            for uploaded in uploaded_files:
                file_path = UPLOAD_DIR / uploaded.name
                # Avoid collision if file exists
                if file_path.exists():
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    file_path = UPLOAD_DIR / f"{file_path.stem}_{timestamp}{file_path.suffix}"

                with open(file_path, "wb") as f:
                    f.write(uploaded.getbuffer())

                chunks = extract_texts_from_file(file_path)
                if not chunks:
                    st.warning(f"No text extracted from {uploaded.name}")
                    continue

                # Add to vector store and SQLite history
                indexed_count = vector_store.add_chunks(chunks)
                chat_history.save_document(
                    name=uploaded.name,
                    path=str(file_path),
                    size_bytes=len(uploaded.getbuffer()),
                    chunk_count=indexed_count,
                )
                added_count += 1

            st.session_state.documents = chat_history.list_documents()
            st.success(f"Successfully processed {added_count} document(s)!")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📚 Knowledge Base Files")

docs = st.session_state.documents
if not docs:
    st.sidebar.info("No documents uploaded yet.")
else:
    for doc in docs:
        col1, col2 = st.sidebar.columns([4, 1])
        size_kb = doc["size_bytes"] / 1024.0 if doc["size_bytes"] else 0
        col1.markdown(f"**{doc['name']}**  \n`{size_kb:.1f} KB | {doc['chunk_count']} chunks`")
        if col2.button("🗑️", key=f"del_{doc['name']}"):
            vector_store.delete_by_source(doc["name"])
            chat_history.delete_document(doc["name"])
            st.session_state.documents = chat_history.list_documents()
            st.sidebar.success(f"Removed {doc['name']}")
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings & Control")

# Backend Selection
backend_choice = st.sidebar.selectbox(
    "LLM Backend",
    options=["local", "openai", "gemini", "ollama"],
    index=["local", "openai", "gemini", "ollama"].index(LLM_BACKEND) if LLM_BACKEND in ["local", "openai", "gemini", "ollama"] else 0,
    help="Select LLM provider. 'local' works offline without API keys.",
)

if backend_choice == "openai" and not OPENAI_API_KEY:
    st.sidebar.warning("⚠️ OPENAI_API_KEY not set in .env. Falling back to Local Grounded mode.")
elif backend_choice == "gemini" and not GEMINI_API_KEY:
    st.sidebar.warning("⚠️ GEMINI_API_KEY not set in .env. Falling back to Local Grounded mode.")


# Mode & Style Toggles
answer_only_docs = st.sidebar.checkbox(
    "Answer ONLY from documents",
    value=True,
    help="If enabled, refuses to answer questions not covered by the documents.",
)

answer_length_choice = st.sidebar.radio(
    "Answer Style",
    options=["Short", "Medium", "Detailed"],
    index=1,
    horizontal=True,
)

if st.sidebar.button("🧹 Clear Chat History", use_container_width=True):
    chat_history.clear_messages()
    st.session_state.messages = []
    st.rerun()

# MAIN VIEW: RAG Chat Assistant
st.markdown(
    """
    <div class="main-header">
        <h1>🤖 RAG Document Assistant</h1>
        <p>Upload PDFs, DOCX, or TXT study materials and get precise, context-grounded answers with exact citations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📌 Source Citations & References"):
                for src in msg["sources"]:
                    page_str = f" (Page {src['page']})" if src.get("page") else ""
                    st.markdown(
                        f"**{src['source']}**{page_str} — *Relevance: {src['similarity']*100:.1f}%*  \n"
                        f"> _{src['snippet']}_"
                    )

# Chat Input & Reaction
query = st.chat_input("Ask a question about your uploaded documents...")
if query:
    # 1. Display user query in UI
    st.session_state.messages.append({"role": "user", "content": query, "sources": None})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Process Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching document vectors & generating answer..."):
            docs_in_kb = chat_history.list_documents()

            if not docs_in_kb:
                answer = "No documents found in knowledge base. Please upload at least one PDF/DOCX/TXT file in the sidebar first."
                sources = []
            else:
                # Retrieve relevant chunks from vector store
                hits = retriever.retrieve(
                    query=query,
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
                        question=query,
                        context=context_str,
                        answer_only_from_docs=answer_only_docs,
                        sources=sources,
                        answer_length=answer_length_choice,
                        backend=backend_choice,
                    )

            # Display response in Streamlit UI
            st.markdown(answer)

            if sources:
                with st.expander("📌 Source Citations & References", expanded=False):
                    for src in sources:
                        page_str = f" (Page {src['page']})" if src.get("page") else ""
                        st.markdown(
                            f"**{src['source']}**{page_str} — *Relevance: {src['similarity']*100:.1f}%*  \n"
                            f"> _{src['snippet']}_"
                        )

            # 3. Save assistant message and sources to SQLite history
            chat_history.save_message(role="user", content=query)
            chat_history.save_message(role="assistant", content=answer, sources=sources)
            st.session_state.messages = chat_history.get_messages()
