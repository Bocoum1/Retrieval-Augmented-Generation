from typing import List, Dict
from rag.embeddings import embed_query
from rag.schemas import Chunk
from rag.vector_store import VectorStore
from rag.bm25 import BM25Store


def reciprocal_rank_fusion(vector_hits, bm25_hits, k: int = 60):
    fused = {}

    for rank, (idx, _) in enumerate(vector_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)

    for rank, (idx, _) in enumerate(bm25_hits, start=1):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store):
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def retrieve(self, query: str, top_k_vector=10, top_k_bm25=10, top_k_fused=10) -> List[Dict]:
        q_emb = embed_query(query)

        vector_hits = self.vector_store.search(q_emb, top_k=top_k_vector)
        bm25_hits = self.bm25_store.search(query, top_k=top_k_bm25)

        fused = reciprocal_rank_fusion(vector_hits, bm25_hits)[:top_k_fused]

        results = []
        for idx, fusion_score in fused:
            chunk: Chunk = self.vector_store.chunks[idx]
            results.append({
                "idx": idx,
                "fusion_score": float(fusion_score),
                "chunk": chunk,
            })

        return results