from typing import List, Tuple
from rag.config import settings


def build_context(reranked_docs: List[dict]) -> Tuple[str, List[str]]:
    parts = []
    citations = []

    total_chars = 0
    for i, item in enumerate(reranked_docs, start=1):
        chunk = item["chunk"]
        citation = f"[{i}] {chunk.source}" + (f", page {chunk.page}" if chunk.page else "")
        block = (
            f"Source {i}\n"
            f"File: {chunk.source}\n"
            f"Page: {chunk.page}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Content:\n{chunk.content}\n"
        )

        if total_chars + len(block) > settings.max_context_chars:
            break

        total_chars += len(block)
        parts.append(block)
        citations.append(citation)

    return "\n\n".join(parts), citations


def build_prompt(question: str, context: str, history: str) -> str:
    return f"""
Tu es un assistant RAG précis.

Règles :
- Réponds uniquement avec les informations présentes dans le contexte.
- Si l'information n'est pas dans le contexte, dis exactement :
  "Je ne trouve pas cette information dans les documents fournis."
- Cite les sources sous la forme [1], [2] si possible.
- N'invente rien.
- Si plusieurs sources semblent se contredire, signale-le.

Historique :
{history}

Contexte :
{context}

Question :
{question}

Réponse en français :
""".strip()