from sentence_transformers import SentenceTransformer, CrossEncoder
from rag.config import settings

_embedder = None
_reranker = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embed_model_name)
    return _embedder


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model_name)
    return _reranker


def embed_texts(texts):
    model = get_embedder()
    return model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def embed_query(query: str):
    return embed_texts([query])