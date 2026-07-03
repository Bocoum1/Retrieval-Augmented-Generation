# Retrieval-Augmented Generation

A local Streamlit application for asking questions over PDF, DOCX, and TXT documents with hybrid retrieval, reranking, citations, and an Ollama-backed language model.

The project is intentionally built as a readable RAG pipeline rather than a black-box demo. Each stage is separated into a small module so the retrieval flow can be inspected, tuned, and extended.

## What it does

- Upload one or more documents from the browser.
- Extract text from PDF, DOCX, and TXT files.
- Split documents into overlapping chunks.
- Build a vector index with sentence-transformer embeddings and FAISS.
- Build a lexical BM25 index over the same chunks.
- Fuse vector and BM25 results with Reciprocal Rank Fusion.
- Rerank candidates with a cross-encoder.
- Generate an answer with a local Ollama model.
- Return citations and expose a diagnostic view of retrieved/reranked chunks.

## Architecture

```text
Documents
  -> loaders.py
  -> chunking.py
  -> vector_store.py + bm25.py
  -> retriever.py
  -> reranker.py
  -> prompting.py
  -> generator.py
  -> Streamlit chat UI
```

The app keeps the retrieval layer transparent: after each answer, the Streamlit UI can show the chunks used, their sources, and their fusion/rerank scores.

## Stack

- Python
- Streamlit
- Sentence Transformers
- FAISS
- BM25
- Cross-encoder reranking
- Ollama
- pypdf
- python-docx

## Repository Structure

```text
.
├── app.py                 # Streamlit interface
├── rag/
│   ├── bm25.py            # Lexical retrieval
│   ├── chunking.py        # Chunk generation
│   ├── config.py          # Runtime settings
│   ├── embeddings.py      # Embedding and reranker loading
│   ├── generator.py       # Ollama generation client
│   ├── loaders.py         # PDF/DOCX/TXT parsing
│   ├── memory.py          # Short chat history formatting
│   ├── pipeline.py        # End-to-end RAG flow
│   ├── prompting.py       # Prompt and citation context
│   ├── reranker.py        # Cross-encoder reranking
│   ├── retriever.py       # Hybrid retrieval and fusion
│   ├── schemas.py         # Shared data structures
│   └── vector_store.py    # FAISS vector search
├── utils/
└── requirements.txt
```

## Getting Started

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and start Ollama, then pull the default model:

```bash
ollama pull phi3
ollama serve
```

Run the app:

```bash
streamlit run app.py
```

Open the local Streamlit URL, upload documents, and ask questions about their content.

## Configuration

Default settings live in `rag/config.py`:

- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Ollama model: `phi3`
- chunk size: `900`
- chunk overlap: `150`
- retrieval top-k values for vector, BM25, fusion, and reranking

The Streamlit sidebar exposes the main retrieval parameters at runtime.

## Notes

This is a local-first RAG prototype. It is useful for understanding and testing the retrieval pipeline, but it does not include production concerns such as authentication, persistent multi-user indexes, background ingestion, or observability.

## Roadmap

- Add example documents and screenshots.
- Add automated tests for chunking, retrieval, and citation formatting.
- Persist indexes between sessions.
- Add evaluation examples for retrieval quality.
- Add streaming responses from Ollama.
