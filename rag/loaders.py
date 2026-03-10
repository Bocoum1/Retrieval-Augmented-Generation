from io import BytesIO
from typing import List
from pypdf import PdfReader
from docx import Document
from rag.schemas import DocumentUnit


def load_pdf(file) -> List[DocumentUnit]:
    reader = PdfReader(BytesIO(file.read()))
    pages = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                DocumentUnit(
                    text=text,
                    source=file.name,
                    page=i,
                    doc_type="pdf"
                )
            )
    return pages


def load_docx(file) -> List[DocumentUnit]:
    doc = Document(BytesIO(file.read()))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not text.strip():
        return []

    return [
        DocumentUnit(
            text=text,
            source=file.name,
            page=None,
            doc_type="docx"
        )
    ]


def load_txt(file) -> List[DocumentUnit]:
    text = file.read().decode("utf-8", errors="ignore")
    if not text.strip():
        return []

    return [
        DocumentUnit(
            text=text,
            source=file.name,
            page=None,
            doc_type="txt"
        )
    ]


def load_document(file) -> List[DocumentUnit]:
    if file.type == "application/pdf":
        return load_pdf(file)

    if file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return load_docx(file)

    if file.type == "text/plain":
        return load_txt(file)

    return []