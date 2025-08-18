from __future__ import annotations

import os

import pytest
from ai_interviewer_pm.retrieval.vectorstore import build_vectorstore, similarity_search


@pytest.mark.skipif(
    os.getenv("RUN_QDRANT_TESTS") != "1" or os.getenv("QDRANT_URL") is None,
    reason="Set RUN_QDRANT_TESTS=1 and QDRANT_URL to run this test.",
)
def test_qdrant_roundtrip() -> None:
    texts = ["hello world", "goodbye moon", "product manager interview"]
    metas = [{"id": i} for i in range(len(texts))]
    store = build_vectorstore(texts, metas)
    res = similarity_search(store, "interview")
    assert len(res) > 0
