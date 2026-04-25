import streamlit as st
import tempfile
import os
from src.document_loader import load_document
from src.chunker import chunk_documents
from src.embeddings import embed_texts
from src.vector_store import add_chunks, has_index, clear_store
from src.rag_chain import answer_question

st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("Document Q&A")

# --- Sidebar: upload & process documents ---
with st.sidebar:
    st.header("Documents")
    files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Process Documents") and files:
        all_chunks = []
        with st.spinner("Processing..."):
            for file in files:
                suffix = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                pages = load_document(tmp_path)
                chunks = chunk_documents(pages)
                all_chunks.extend(chunks)
                os.unlink(tmp_path)

            if all_chunks:
                texts = [c["text"] for c in all_chunks]
                embeddings = embed_texts(texts)
                add_chunks(all_chunks, embeddings)

        st.success(f"Done! {len(all_chunks)} chunks from {len(files)} file(s).")

    if st.button("Clear All Documents"):
        clear_store()
        st.success("Cleared.")

# --- Main area: ask questions ---
question = st.text_input("Ask a question about your documents:")

if question:
    if not has_index():
        st.warning("Upload and process documents first.")
    else:
        with st.spinner("Thinking..."):
            answer, sources = answer_question(question)

        st.markdown("### Answer")
        st.write(answer)

        if sources:
            with st.expander("Retrieved Sources"):
                for i, chunk in enumerate(sources, 1):
                    st.markdown(f"**Source {i}: {chunk['source']}, Page {chunk['page']}**")
                    st.text(chunk["text"][:300])
                    st.divider()
