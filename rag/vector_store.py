import json
from pathlib import Path
from typing import List, Tuple
import faiss
import numpy as np
from rag.embeddings import embed_texts
from rag.schemas import Chunk


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.content for c in chunks]

        embeddings = embed_texts(texts).astype("float32")
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        scores, indices = self.index.search(query_embedding.astype("float32"), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((int(idx), float(score)))
        return results

    def save(self, index_path: Path, meta_path: Path) -> None:
        if self.index is None:
            raise ValueError("Index non initialisé")

        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=2)

    def load(self, index_path: Path, meta_path: Path) -> None:
        self.index = faiss.read_index(str(index_path))

        from rag.schemas import Chunk
        import json

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = [Chunk(**item) for item in data]