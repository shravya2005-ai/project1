import os
import streamlit as st
from pathlib import Path
from datetime import datetime

from config import UPLOAD_DIR, OPENAI_API_KEY, LLM_BACKEND
from src.chat_history import ChatHistory
from src.document_processor import extract_texts_from_file, SUPPORTED_EXTENSIONS
from src.vector_store import VectorStore
from src.llm import answer_question

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📄",
    layout="wide",
)

if LLM_BACKEND == "openai" and not OPENAI_API_KEY:
    st.sidebar.error("Set OPENAI_API_KEY in .env before using the app, or switch to LLM_BACKEND=local.")
else:
    st.sidebar.info(f"LLM backend: {LLM_BACKEND.upper()}")

storage = VectorStore()
history = ChatHistory()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

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

st.title("RAG-Based Document Assistant")
st.write(
    "Upload documents, then ask questions about their contents. The assistant retrieves the best document excerpts and answers using OpenAI."
)

question = st.text_input("Ask a question", key="question_input")
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
            )
            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Source excerpts")
            for item in results:
                metadata = item["metadata"]
                st.write(
                    f"**{metadata.get('source')}" 
                    + (f" – page {metadata.get('page')}" if metadata.get("page") else "")
                )
                st.write(item["document"])

            st.session_state.conversation.append({"question": question, "answer": answer, "sources": list(dict.fromkeys(sources))})
            history.save_message("user", question)
            history.save_message("assistant", answer)

if st.session_state.conversation:
    st.markdown("---")
    st.subheader("Conversation history")
    for turn in st.session_state.conversation[::-1]:
        st.markdown(f"**Q:** {turn['question']}")
        st.markdown(f"**A:** {turn['answer']}")
        st.markdown(f"**Sources:** {', '.join(turn['sources'])}")
