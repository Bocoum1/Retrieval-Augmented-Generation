from typing import List, Dict
from rag.embeddings import get_reranker


def rerank(query: str, candidates: List[Dict], top_k: int = 4) -> List[Dict]:
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [[query, item["chunk"].content] for item in candidates]
    scores = reranker.predict(pairs)

    reranked = []
    for score, item in zip(scores, candidates):
        row = item.copy()
        row["rerank_score"] = float(score)
        reranked.append(row)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]