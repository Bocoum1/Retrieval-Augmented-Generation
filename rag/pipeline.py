from rag.config import settings
from rag.prompting import build_context, build_prompt
from rag.generator import generate_answer
from rag.reranker import rerank


def answer_query(query: str, retriever, history: str):
    retrieved = retriever.retrieve(
        query=query,
        top_k_vector=settings.top_k_vector,
        top_k_bm25=settings.top_k_bm25,
        top_k_fused=settings.top_k_fused,
    )

    reranked = rerank(
        query=query,
        candidates=retrieved,
        top_k=settings.top_k_rerank
    )

    context, citations = build_context(reranked)
    prompt = build_prompt(query, context, history)
    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "citations": citations,
        "retrieved": retrieved,
        "reranked": reranked,
        "context": context,
        "prompt": prompt,
    }