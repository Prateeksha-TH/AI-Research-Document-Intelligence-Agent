# AI Research & Document Intelligence Agent

A multi-document RAG (Retrieval-Augmented Generation) application that lets users upload PDF research material, retrieve semantically relevant evidence, and generate source-grounded answers.

## Highlights

- Upload and index multiple text-based PDFs
- Extract and clean text with PyMuPDF
- Character-based chunking with overlap
- Generate semantic embeddings with Sentence Transformers
- Persist embeddings in ChromaDB
- Retrieve top-k evidence using cosine similarity
- Generate answers with an OpenAI-compatible LLM
- Preserve document + page metadata for traceability
- Retrieval-only mode when no LLM API key is configured
- Streamlit user interface
- Unit tests + GitHub Actions CI

## Architecture

```text
PDFs
  ↓
PyMuPDF extraction
  ↓
Cleaning + overlapping chunks
  ↓
Sentence Transformer embeddings
  ↓
ChromaDB vector store
  ↓
Top-k semantic retrieval ← User question
  ↓
Evidence-grounded context
  ↓
Optional LLM generation
  ↓
Answer + retrieved source evidence
```

## Project structure

```text
AI-Research-Document-Intelligence-Agent/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document_processor.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── llm.py
├── tests/
│   ├── test_chunking.py
│   └── test_retrieval.py
├── data/
│   └── .gitkeep
└── .github/
    └── workflows/
        └── tests.yml
```

## Tech stack

**Python · Streamlit · ChromaDB · Sentence Transformers · PyMuPDF · OpenAI-compatible API · Pytest · GitHub Actions**

## Run locally

### 1. Clone

```bash
git clone https://github.com/Prateeksha-TH/AI-Research-Document-Intelligence-Agent.git
cd AI-Research-Document-Intelligence-Agent
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the optional LLM

Copy `.env.example` to `.env`.

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=
```

The application still supports semantic retrieval without an API key; generated answers require a configured OpenAI-compatible API.

### 5. Start the application

```bash
streamlit run app.py
```

Upload one or more PDFs, click **Index documents**, and ask questions about the indexed material.

## Testing

```bash
pytest -q
```

The current test suite covers chunking behavior and source-aware context construction.

## Engineering decisions

### Evidence before generation

The application retrieves relevant chunks before sending context to the LLM. This keeps the retrieval step inspectable and preserves source metadata.

### Grounded prompting

The LLM is instructed to use only the supplied evidence, cite `[Source N]` references, and acknowledge when the retrieved material is insufficient.

### Deterministic vector IDs

Chunk IDs are derived from document metadata and content, allowing ChromaDB to safely upsert repeated indexing operations.

### Modular design

Document processing, vector storage, retrieval, and generation are separated into modules so components can be tested or replaced independently.

## Limitations

- Scanned/image-only PDFs are not OCR'd in the current version.
- The first embedding-model startup downloads model weights.
- Retrieval quality depends on document quality, chunking, and embedding relevance.
- LLM generation requires an API key and may incur provider costs.

## Roadmap

- [x] PDF ingestion
- [x] Semantic retrieval
- [x] Source/page metadata
- [x] Grounded generation
- [x] Retrieval-only mode
- [x] Automated tests
- [x] GitHub Actions CI
- [ ] Hybrid keyword + semantic retrieval
- [ ] Retrieval evaluation dataset
- [ ] RAG evaluation metrics
- [ ] OCR for scanned PDFs
- [ ] Cross-document comparison mode
- [ ] Deployment configuration

## Resume-ready description

**AI Research & Document Intelligence Agent** — Built a multi-document RAG system using Python, ChromaDB, Sentence Transformers and an OpenAI-compatible LLM to retrieve relevant PDF evidence and generate source-grounded research answers with document/page traceability.
