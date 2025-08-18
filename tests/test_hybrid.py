from __future__ import annotations

from ai_interviewer_pm.retrieval.hybrid import BM25Retriever, hybrid_rerank


def test_bm25_and_hybrid() -> None:
    corpus = [
        "product management interview questions",
        "behavioral interview case study",
        "neural networks and embeddings",
    ]
    bm = BM25Retriever(corpus)
    s = bm.search("interview", k=2)
    assert len(s) == 2
    dense = [(corpus[1], 0.7), (corpus[0], 0.6)]
    fused = hybrid_rerank(dense, s, alpha=0.6)
    assert len(fused) >= 2
