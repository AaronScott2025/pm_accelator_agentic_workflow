from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ai_interviewer_pm.settings import settings
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from qdrant_client import QdrantClient


@dataclass
class VectorDoc:
    """Vectorized document wrapper."""

    text: str
    metadata: dict[str, object]


def get_embedding_model(provider: str = "openai", *, model: str | None = None) -> Embeddings:
    """Factory for embeddings models.

    Args:
        provider: Provider key, e.g., "cohere". More can be added later.
        model: Optional provider-specific model name.

    Returns:
        LangChain Embeddings implementation.
    """
    if provider == "cohere":
        model_name = model or "embed-english-v3.0"
        return CohereEmbeddings(model=model_name, cohere_api_key=settings.cohere_api_key)
    if provider == "openai":
        model_name = model or "text-embedding-3-large"
        # Let langchain-openai read OPENAI_API_KEY from environment
        return OpenAIEmbeddings(model=model_name)
    raise ValueError(f"Unknown embeddings provider: {provider}")


def get_qdrant_client() -> QdrantClient:
    """Create or reuse Qdrant client from settings."""
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def build_vectorstore(
    texts: Sequence[str],
    metadatas: Sequence[dict[str, object]] | None,
    *,
    provider: str = "openai",
    model: str | None = None,
    collection_name: str | None = None,
) -> Qdrant:
    """Index texts into a Qdrant vector store.

    Args:
        texts: Documents to index.
        metadatas: Parallel list of metadata dicts.
        provider: Embeddings provider key.
        model: Optional model name.
        collection_name: Qdrant collection name (defaults to settings).

    Returns:
        LangChain Qdrant vectorstore.
    """
    embeddings = get_embedding_model(provider, model=model)
    client = get_qdrant_client()
    coll = collection_name or settings.qdrant_collection

    # If no texts provided, return a wrapper without creating the collection yet.
    if not texts:
        return Qdrant(client=client, collection_name=coll, embeddings=embeddings)

    return Qdrant.from_texts(
        texts=list(texts),
        metadatas=list(metadatas) if metadatas is not None else None,
        embedding=embeddings,
        client=client,
        collection_name=coll,
    )


def similarity_search(
    store: Qdrant, query: str, *, k: int = 5
) -> list[tuple[str, dict[str, object], float]]:
    """Simple similarity search over a Qdrant vectorstore.

    Returns tuples of (text, metadata, score). Returns an empty list if the
    collection doesn't exist yet or another retriever error occurs.
    """
    try:
        docs = store.similarity_search_with_score(query, k=k)
    except Exception:
        return []
    return [(d.page_content, dict(d.metadata), float(score)) for d, score in docs]
