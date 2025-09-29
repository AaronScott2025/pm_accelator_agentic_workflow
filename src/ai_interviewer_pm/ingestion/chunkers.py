from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)


@dataclass
class Chunk:
    """A single chunk of text with optional metadata.

    Attributes:
        text: The chunk content.
        metadata: Arbitrary metadata map (e.g., source, start/end indices, timestamps).
    """

    text: str
    metadata: dict[str, object]


class BaseChunker:
    """Abstract chunker API."""

    def split(self, text: str, *, metadata: Optional[dict[str, object]] = None) -> List[Chunk]:
        """Split a text into chunks.

        Args:
            text: The input text to split.
            metadata: Optional base metadata added to each chunk.

        Returns:
            List of chunks.
        """
        raise NotImplementedError


class RecursiveCharChunker(BaseChunker):
    """Recursive character splitter with overlap suitable for RAG baselines."""

    def __init__(self, *, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def split(self, text: str, *, metadata: Optional[dict[str, object]] = None) -> List[Chunk]:
        docs = self._splitter.create_documents([text], metadatas=[metadata or {}])
        return [Chunk(d.page_content, dict(d.metadata)) for d in docs]


class SentenceTokenChunker(BaseChunker):
    """Token-based sentence chunker using sentence-transformers token splitter."""

    def __init__(self, *, tokens_per_chunk: int = 256, chunk_overlap: int = 32) -> None:
        self._splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap, tokens_per_chunk=tokens_per_chunk
        )

    def split(self, text: str, *, metadata: Optional[dict[str, object]] = None) -> List[Chunk]:
        chunks = self._splitter.split_text(text)
        base = metadata or {}
        return [Chunk(c, dict(base)) for c in chunks]


class SentenceBoundaryChunker(BaseChunker):
    """Sentence-based chunking using NLTK sentence tokenizer with batching.

    Falls back to a simple period-based splitter if NLTK punkt is unavailable.
    """

    def __init__(self, *, sentences_per_chunk: int = 5) -> None:
        self.sentences_per_chunk = sentences_per_chunk
        # Lazy import to avoid hard NLTK dependency at import time
        try:
            import nltk  # noqa: WPS433

            try:
                nltk.data.find("tokenizers/punkt")
                self._sent_tokenize = nltk.sent_tokenize
            except LookupError:
                # Fallback without downloading
                self._sent_tokenize = lambda t: [s.strip() for s in t.split(".") if s.strip()]
        except Exception:
            self._sent_tokenize = lambda t: [s.strip() for s in t.split(".") if s.strip()]

    def split(self, text: str, *, metadata: Optional[dict[str, object]] = None) -> List[Chunk]:
        sentences = self._sent_tokenize(text)
        chunks: List[Chunk] = []
        base = metadata or {}
        for i in range(0, len(sentences), self.sentences_per_chunk):
            group = " ".join(sentences[i : i + self.sentences_per_chunk])
            chunks.append(Chunk(group, dict(base)))
        return chunks


class TimestampChunker(BaseChunker):
    """Very simple timestamp chunker for VTT-like transcripts.

    Assumes lines with timecodes and accumulates content between them.
    """

    def split(self, text: str, *, metadata: Optional[dict[str, object]] = None) -> List[Chunk]:
        lines = text.splitlines()
        chunks: List[Chunk] = []
        current: list[str] = []
        current_meta: dict[str, object] | None = None
        base = metadata or {}

        def flush() -> None:
            nonlocal current, current_meta
            if current:
                chunks.append(Chunk("\n".join(current).strip(), dict(base | (current_meta or {}))))
                current = []
                current_meta = None

        for line in lines:
            if "-->" in line:
                flush()
                current_meta = {"timecode": line.strip()}
            elif line.strip().isdigit():
                # ignore VTT cue index
                continue
            else:
                current.append(line)
        flush()
        return [c for c in chunks if c.text]


class ParagraphChunker(BaseChunker):
    """Paragraph-based chunker that splits on blank lines and merges small paragraphs."""

    def __init__(self, *, min_chars: int = 200) -> None:
        self.min_chars = min_chars

    def split(self, text: str, *, metadata: Optional[dict[str, object]] = None) -> List[Chunk]:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        merged: list[str] = []
        buf = ""
        for p in paras:
            if len(buf) + len(p) < self.min_chars:
                buf = f"{buf}\n\n{p}".strip()
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)
        base = metadata or {}
        return [Chunk(m, dict(base)) for m in merged]


class SemanticSimilarityChunker(BaseChunker):
    """Chunker that breaks where consecutive sentence embeddings diverge beyond a threshold.

    Requires sentence-transformers; falls back to SentenceBoundaryChunker when unavailable.
    """

    def __init__(self, *, model: str = "all-MiniLM-L6-v2", threshold: float = 0.25) -> None:
        self.model_name = model
        self.threshold = threshold
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(model)
        except Exception:
            self._model = None
        self._fallback = SentenceBoundaryChunker(sentences_per_chunk=5)

    def split(self, text: str, *, metadata: Optional[dict[str, object]] = None) -> List[Chunk]:
        if self._model is None:
            return self._fallback.split(text, metadata=metadata)
        sentences = self._fallback._sent_tokenize(text)  # reuse tokenizer
        if not sentences:
            return []
        import numpy as np  # noqa: WPS433

        embs = self._model.encode(sentences, normalize_embeddings=True)
        breaks = [0]
        for i in range(1, len(sentences)):
            sim = float(np.dot(embs[i - 1], embs[i]))
            if 1 - sim > self.threshold:
                breaks.append(i)
        breaks.append(len(sentences))
        chunks: list[Chunk] = []
        base = metadata or {}
        for s, e in zip(breaks[:-1], breaks[1:]):
            txt = " ".join(sentences[s:e])
            chunks.append(Chunk(txt, dict(base)))
        return chunks


def chunk_texts(
    texts: Sequence[str],
    chunker: BaseChunker,
    *,
    base_metadata: Optional[dict[str, object]] = None,
) -> List[Chunk]:
    """Apply a chunker to a sequence of texts and return all chunks.

    Args:
        texts: Input texts.
        chunker: Chunker implementation.
        base_metadata: Optional metadata added to every chunk.

    Returns:
        List of chunks across inputs.
    """
    out: list[Chunk] = []
    for t in texts:
        out.extend(chunker.split(t, metadata=base_metadata))
    return out
