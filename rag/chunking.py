from typing import List
from rag.schemas import DocumentUnit, Chunk
from rag.config import settings


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def build_chunks(documents: List[DocumentUnit]) -> List[Chunk]:
    chunks = []

    for doc in documents:
        parts = chunk_text(
            text=doc.text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        for pos, content in enumerate(parts):
            chunk_id = f"{doc.source}::{doc.page}::{pos}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    content=content,
                    source=doc.source,
                    page=doc.page,
                    doc_type=doc.doc_type,
                    position=pos,
                )
            )

    return chunks