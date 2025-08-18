from __future__ import annotations

from ai_interviewer_pm.ingestion.chunkers import (
    RecursiveCharChunker,
    SentenceBoundaryChunker,
    SentenceTokenChunker,
    TimestampChunker,
)


def test_recursive_char_chunker_basic() -> None:
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 50
    chunks = RecursiveCharChunker(chunk_size=200, chunk_overlap=20).split(text)
    assert len(chunks) > 1
    assert all(c.text for c in chunks)


def test_sentence_boundary_chunker_batches_sentences() -> None:
    text = "This is a sentence. Here is another. And another. One more. Final sentence."
    chunks = SentenceBoundaryChunker(sentences_per_chunk=2).split(text)
    assert len(chunks) >= 2


def test_sentence_token_chunker() -> None:
    text = "Short text for token splitting. " * 20
    chunks = SentenceTokenChunker(tokens_per_chunk=16).split(text)
    assert len(chunks) > 1


def test_timestamp_chunker_vtt() -> None:
    text = """WEBVTT\n\n1\n00:00:00.000 --> 00:00:05.000\nHello there.\n2\n00:00:05.000 --> 00:00:10.000\nGeneral Kenobi.\n"""
    chunks = TimestampChunker().split(text)
    assert any("timecode" in c.metadata for c in chunks)
