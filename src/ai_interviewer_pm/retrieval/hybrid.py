from __future__ import annotations

from typing import List, Sequence, Tuple, Optional


def rrf_fusion(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]],
    *,
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion (RRF) of multiple ranking lists.

    RRF combines multiple ranked lists into a single ranking by summing
    1 / (k + rank) across lists for each document.

    Args:
        ranked_lists: Sequence of ranked lists where each list contains
            (document, score). Only the ordering matters; scores are ignored.
        k: Smoothing constant controlling contribution of lower-ranked items.

    Returns:
        Fused ranking as a list of (document, rrf_score) sorted descending.
    """
    # Build mapping: doc -> cumulative RRF score
    rrf_scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for idx, (doc, _score) in enumerate(ranked):
            contrib = 1.0 / (k + (idx + 1))
            rrf_scores[doc] = rrf_scores.get(doc, 0.0) + contrib
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Lightweight BM25 retriever for baseline comparisons.

    Handles empty corpora gracefully (returns empty results).
    """

    def __init__(self, corpus: Sequence[str]) -> None:
        self._corpus = list(corpus)
        if len(self._corpus) == 0:
            self._docs: list[list[str]] = []
            self._bm25: Optional[BM25Okapi] = None
        else:
            self._docs = [d.split() for d in self._corpus]
            self._bm25 = BM25Okapi(self._docs)

    def search(self, query: str, *, k: int = 5) -> List[Tuple[str, float]]:
        if not self._corpus or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.split())
        ranked = sorted(zip(self._corpus, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]


def hybrid_rerank(
    dense_hits: Sequence[Tuple[str, float]],
    sparse_hits: Sequence[Tuple[str, float]],
    *,
    alpha: float = 0.5,
) -> List[Tuple[str, float]]:
    """Linear fusion of dense (vector) and sparse (BM25) scores.

    Args:
        dense_hits: Pairs of (doc, score) from vector search.
        sparse_hits: Pairs of (doc, score) from BM25.
        alpha: Weight for dense scores.

    Returns:
        Reranked list of (doc, fused_score).
    """
    dmap = {d: s for d, s in dense_hits}
    smap = {d: s for d, s in sparse_hits}
    keys = set(dmap) | set(smap)
    out = []
    for k in keys:
        ds = dmap.get(k, 0.0)
        ss = smap.get(k, 0.0)
        out.append((k, alpha * ds + (1 - alpha) * ss))
    return sorted(out, key=lambda x: x[1], reverse=True)
