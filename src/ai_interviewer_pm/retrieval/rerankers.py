from __future__ import annotations

from typing import List, Sequence, Tuple


def cohere_rerank(query: str, docs: Sequence[str], *, top_n: int = 5) -> List[Tuple[str, float]]:
    """Use Cohere Rerank (v3 default) to rerank documents.

    Returns list of (doc, score) sorted by score desc.
    """
    try:
        import cohere  # type: ignore
        from ai_interviewer_pm.settings import settings

        client = cohere.Client(api_key=settings.cohere_api_key)
        res = client.rerank(query=query, documents=list(docs), top_n=min(top_n, len(docs)))
        # API shape may differ; adapt fields accordingly
        pairs = [
            (r.document.get("text", str(docs[r.index])), float(r.relevance_score)) for r in res
        ]
        return sorted(pairs, key=lambda x: x[1], reverse=True)
    except Exception:
        # Fallback: return original order with neutral scores
        return [(d, 0.0) for d in docs[:top_n]]


def cross_encoder_rerank(
    query: str,
    docs: Sequence[str],
    *,
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """Use sentence-transformers CrossEncoder to rerank.

    Requires sentence-transformers; falls back to input order on failure.
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        ce = CrossEncoder(model)
        pairs = [[query, d] for d in docs]
        scores = ce.predict(pairs)
        ranked = sorted(zip(docs, [float(s) for s in scores]), key=lambda x: x[1], reverse=True)
        return ranked[:top_n]
    except Exception:
        return [(d, 0.0) for d in docs[:top_n]]
