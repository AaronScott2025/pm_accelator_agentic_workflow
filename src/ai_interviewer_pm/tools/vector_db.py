from __future__ import annotations

from typing import List

from ai_interviewer_pm.retrieval.vectorstore import get_embedding_model, get_qdrant_client
from ai_interviewer_pm.settings import settings
from langchain_community.vectorstores import Qdrant


def vector_search(query: str, *, k: int = 5) -> List[dict]:
    """Search the Qdrant vector DB and return top-k results.

    Args:
        query: Query string.
        k: Number of results to return.

    Returns:
        List of dicts with text, metadata, and score.
    """
    client = get_qdrant_client()
    embeddings = get_embedding_model()
    store = Qdrant(client=client, collection_name=settings.qdrant_collection, embeddings=embeddings)
    try:
        docs = store.similarity_search_with_score(query, k=k)
    except Exception:
        # Collection may not exist yet; return empty results gracefully
        return []
    return [
        {"text": d.page_content, "metadata": dict(d.metadata), "score": float(score)}
        for d, score in docs
    ]
