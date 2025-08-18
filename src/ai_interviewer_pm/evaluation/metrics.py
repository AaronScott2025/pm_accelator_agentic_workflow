from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class RetrievalEvalResult:
    """Holds evaluation results for a single configuration."""

    name: str
    mrr: float
    recall_at_k: dict[int, float]


def mean_reciprocal_rank(ranked_lists: Sequence[Sequence[int]], truths: Sequence[int]) -> float:
    """Compute MRR given ranked doc id lists and ground truth id per query.

    Args:
        ranked_lists: For each query, a list of doc ids sorted by descending score.
        truths: The ground truth relevant doc id per query.

    Returns:
        MRR value in [0, 1].
    """
    acc = 0.0
    for ranks, truth in zip(ranked_lists, truths):
        rr = 0.0
        for idx, doc_id in enumerate(ranks, start=1):
            if doc_id == truth:
                rr = 1.0 / idx
                break
        acc += rr
    return acc / max(1, len(ranked_lists))


def recall_at_k(
    ranked_lists: Sequence[Sequence[int]],
    truths: Sequence[int],
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[int, float]:
    """Compute recall@k across a set of queries.

    Returns a dict mapping k to recall.
    """
    out: dict[int, float] = {}
    for k in k_values:
        hits = 0
        for ranks, truth in zip(ranked_lists, truths):
            if truth in ranks[:k]:
                hits += 1
        out[k] = hits / max(1, len(ranked_lists))
    return out
