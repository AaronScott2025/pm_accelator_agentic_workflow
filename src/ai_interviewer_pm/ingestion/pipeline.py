from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from ai_interviewer_pm.ingestion.chunkers import (
    BaseChunker,
    RecursiveCharChunker,
    SentenceBoundaryChunker,
    TimestampChunker,
)
from ai_interviewer_pm.retrieval.vectorstore import build_vectorstore
from ai_interviewer_pm.settings import settings


def load_texts_from_data_dir(data_dir: str | Path | None = None) -> List[Tuple[str, dict]]:
    """Load raw texts from data directory with minimal heuristics.

    Currently supports .vtt; future: .txt, .md, .pdf with OCR.

    Args:
        data_dir: Directory to scan (defaults to Settings.data_dir).

    Returns:
        List of (text, metadata) tuples.
    """
    root = Path(data_dir or settings.data_dir)
    out: list[Tuple[str, dict]] = []
    for p in root.glob("**/*"):
        if p.suffix.lower() == ".vtt":
            out.append((p.read_text(encoding="utf-8"), {"source": str(p)}))
        # Add more parsers here as needed
    return out


def pick_chunker(name: str) -> BaseChunker:
    """Factory for configured chunkers by short name."""
    if name == "recursive":
        return RecursiveCharChunker(chunk_size=800, chunk_overlap=160)
    if name == "sentence":
        return SentenceBoundaryChunker(sentences_per_chunk=6)
    if name == "timestamp":
        return TimestampChunker()
    raise ValueError(f"Unknown chunker: {name}")


def index_data(
    *,
    chunker_name: str = "timestamp",
    provider: str = "cohere",
    model: str | None = None,
    collection_name: str | None = None,
):
    """End-to-end ingestion: load data, chunk, and index into Qdrant.

    Args:
        chunker_name: One of {'recursive','sentence','timestamp'}.
        provider: Embeddings provider key ('cohere' supported).
        model: Optional embedding model name.
        collection_name: Optional Qdrant collection name.

    Returns:
        The created vectorstore object.
    """
    raw = load_texts_from_data_dir()
    chunker = pick_chunker(chunker_name)

    chunks: list[str] = []
    metas: list[dict] = []
    for text, meta in raw:
        idx = 0
        for c in chunker.split(text, metadata=meta | {"chunker": chunker_name, "chunk_index": idx}):
            chunks.append(c.text)
            metas.append(c.metadata)
            idx += 1

    if not chunks:
        # Nothing to index; return None gracefully
        print("No data found to index in data/. Skipping Qdrant indexing.")
        return None

    store = build_vectorstore(
        texts=chunks,
        metadatas=metas,
        provider=provider,
        model=model,
        collection_name=collection_name,
    )
    return store
