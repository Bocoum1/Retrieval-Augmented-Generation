from typing import List, Tuple
from rank_bm25 import BM25Okapi
from rag.schemas import Chunk


def tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Store:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            list(enumerate(scores)),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:top_k]