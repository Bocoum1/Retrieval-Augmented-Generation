import hashlib

import streamlit as st

from rag.config import settings
from rag.loaders import load_document
from rag.chunking import build_chunks
from rag.vector_store import VectorStore
from rag.bm25 import BM25Store
from rag.retriever import HybridRetriever
from rag.pipeline import answer_query
from rag.memory import append_history


st.set_page_config(page_title="Assistant RAG local", page_icon="📚", layout="wide")
st.title("Assistant RAG local")
st.caption("PDF / DOCX / TXT · recherche hybride · reranking · citations · Ollama")


def empty_rag_state():
    return {
        "vector_store": None,
        "bm25_store": None,
        "retriever": None,
        "files_signature": None,
        "chunks_count": 0,
    }


def uploaded_files_signature(files):
    signature = []
    for file in files:
        data = file.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        signature.append((file.name, file.type, len(data), digest))
    return tuple(signature)


if "history" not in st.session_state:
    st.session_state.history = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "rag_state" not in st.session_state:
    st.session_state.rag_state = empty_rag_state()

state = st.session_state.rag_state


with st.sidebar:
    st.header("Réglages")
    settings.ollama_model = st.text_input("Modèle Ollama", value=settings.ollama_model)
    settings.top_k_vector = st.slider("Top-k vector", 2, 20, settings.top_k_vector)
    settings.top_k_bm25 = st.slider("Top-k BM25", 2, 20, settings.top_k_bm25)
    settings.top_k_fused = st.slider("Top-k fusion", 2, 20, settings.top_k_fused)
    settings.top_k_rerank = st.slider("Top-k rerank", 1, 10, settings.top_k_rerank)

    if st.button("Réinitialiser"):
        for key in ["history", "messages", "indexed_files", "rag_state"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


uploaded_files = st.file_uploader(
    "Ajoute un ou plusieurs documents, puis pose une question",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    current_signature = uploaded_files_signature(uploaded_files)

    if state["files_signature"] != current_signature:
        documents = []
        for file in uploaded_files:
            try:
                file.seek(0)
                file_docs = load_document(file)
                documents.extend(file_docs)
            except Exception as e:
                st.error(f"Erreur sur {file.name}: {e}")

        chunks = build_chunks(documents)
        if chunks:
            vector_store = VectorStore()
            vector_store.build(chunks)

            bm25_store = BM25Store(chunks)
            retriever = HybridRetriever(vector_store, bm25_store)

            state["vector_store"] = vector_store
            state["bm25_store"] = bm25_store
            state["retriever"] = retriever
            state["files_signature"] = current_signature
            state["chunks_count"] = len(chunks)
            st.session_state.indexed_files = [f.name for f in uploaded_files]
            st.success(f"{len(uploaded_files)} document(s) indexé(s), {len(chunks)} chunks générés.")
        else:
            state.update(empty_rag_state())
            state["files_signature"] = current_signature
            st.session_state.indexed_files = []
            st.warning("Aucun texte exploitable n'a été trouvé dans les documents importés.")
    else:
        st.info(
            f"{len(st.session_state.indexed_files)} document(s) déjà indexé(s), "
            f"{state['chunks_count']} chunks disponibles."
        )

if st.session_state.indexed_files:
    with st.expander("Fichiers indexés"):
        for name in st.session_state.indexed_files:
            st.write(f"- {name}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


query = st.chat_input("Pose une question sur les documents")

if query:
    with st.chat_message("user"):
        st.write(query)

    st.session_state.messages.append({"role": "user", "content": query})

    if state["retriever"] is None:
        answer = "Ajoute d'abord un ou plusieurs documents."
        with st.chat_message("assistant"):
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        try:
            result = answer_query(
                query=query,
                retriever=state["retriever"],
                history=st.session_state.history
            )

            final_answer = result["answer"]
            if result["citations"]:
                final_answer += "\n\n**Sources**\n"
                final_answer += "\n".join(f"- {c}" for c in result["citations"])

            with st.chat_message("assistant"):
                st.write(final_answer)

                with st.expander("Diagnostic RAG"):
                    st.markdown("**Sources rerankées**")
                    for i, item in enumerate(result["reranked"], start=1):
                        chunk = item["chunk"]
                        st.markdown(
                            f"**[{i}] {chunk.source}**"
                            + (f" — page {chunk.page}" if chunk.page else "")
                        )
                        st.caption(
                            f"fusion_score={item.get('fusion_score', 0):.4f} • "
                            f"rerank_score={item.get('rerank_score', 0):.4f}"
                        )
                        st.write(chunk.content)

            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.session_state.history = append_history(
                st.session_state.history,
                query,
                result["answer"]
            )

        except Exception as e:
            err = f"Erreur : {e}"
            with st.chat_message("assistant"):
                st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
