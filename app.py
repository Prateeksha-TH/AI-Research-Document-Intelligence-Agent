import tempfile
from pathlib import Path

import streamlit as st

from document_processor import extract_pdf_chunks
from llm import generate_answer
from retriever import retrieve
from vector_store import VectorStore


st.set_page_config(
    page_title="AI Research & Document Intelligence Agent",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Research & Document Intelligence Agent")
st.caption(
    "Multi-document RAG assistant for semantic retrieval and source-grounded research."
)


@st.cache_resource
def get_store():
    return VectorStore()


store = get_store()

with st.sidebar:
    st.header("Document Index")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Index documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            total_chunks = 0
            with st.spinner("Extracting, chunking and embedding documents..."):
                for uploaded in uploaded_files:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = Path(tmp.name)

                    try:
                        chunks = extract_pdf_chunks(tmp_path)
                        for chunk in chunks:
                            chunk.source = uploaded.name
                        total_chunks += store.add_chunks(chunks)
                    finally:
                        tmp_path.unlink(missing_ok=True)

            st.success(f"Indexed {total_chunks} chunks.")

    if st.button("Clear index", use_container_width=True):
        store.reset()
        st.success("Vector index cleared.")

    st.metric("Indexed chunks", store.count())

question = st.text_area(
    "Ask a question about your uploaded documents",
    placeholder="Example: What are the main findings discussed across the documents?",
    height=120,
)

top_k = st.slider("Retrieved evidence chunks", 2, 10, 5)

if st.button("Research", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving evidence..."):
        hits = retrieve(store, question, top_k=top_k)

    if not hits:
        st.info("No indexed evidence is available. Upload and index documents first.")
    else:
        st.subheader("Answer")
        with st.spinner("Generating source-grounded response..."):
            answer = generate_answer(question, hits)
        st.write(answer)

        st.subheader("Retrieved evidence")
        for i, hit in enumerate(hits, start=1):
            with st.expander(
                f"[Source {i}] {hit['source']} — page {hit['page']}"
            ):
                st.write(hit["text"])
                st.caption(f"Cosine distance: {hit['distance']:.4f}")

st.divider()
st.caption(
    "Evidence-first design: retrieval happens before generation, and source metadata "
    "is preserved for traceability."
)
