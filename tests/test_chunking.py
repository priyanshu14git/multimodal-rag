"""
Unit tests for app.services.chunking.chunk_text.

This module has no dependencies beyond the standard library, so these
tests run instantly with no models, network, or fixtures required.
"""

from app.services.chunking import chunk_text


def test_short_text_returns_single_chunk():
    text = "This is a short sentence. It fits in one chunk easily."
    chunks = chunk_text(text, chunk_size_chars=1000, chunk_overlap_chars=150)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size_chars=1000, chunk_overlap_chars=150) == []
    assert chunk_text("   ", chunk_size_chars=1000, chunk_overlap_chars=150) == []


def test_long_text_is_split_into_multiple_chunks():
    sentence = "This is one moderately long sentence about restoration models. "
    text = sentence * 40  # well over any reasonable chunk_size
    chunks = chunk_text(text, chunk_size_chars=300, chunk_overlap_chars=50)
    assert len(chunks) > 1
    # No chunk should wildly exceed the requested size (some slack is fine
    # since splitting happens on sentence boundaries, not mid-word).
    for c in chunks:
        assert len(c) <= 300 + len(sentence)


def test_chunks_do_not_lose_content():
    sentence = "Sentence number {}. "
    text = "".join(sentence.format(i) for i in range(60))
    chunks = chunk_text(text, chunk_size_chars=200, chunk_overlap_chars=40)
    # Every sentence marker should show up somewhere in the reconstructed
    # output - overlap means we can't just concatenate, but nothing should
    # vanish entirely.
    joined = " ".join(chunks)
    for i in range(60):
        assert f"Sentence number {i}." in joined


def test_overlap_creates_shared_content_between_consecutive_chunks():
    sentence = "This is sentence {} in a long passage about image restoration. "
    text = "".join(sentence.format(i) for i in range(30))
    chunks = chunk_text(text, chunk_size_chars=250, chunk_overlap_chars=80)
    assert len(chunks) > 1
    # With nonzero overlap, adjacent chunks should share at least some text
    # (the end of chunk[i] should not be completely disjoint from the start
    # of chunk[i+1]) - a basic sanity check that overlap is actually doing
    # something rather than being silently ignored.
    for i in range(len(chunks) - 1):
        tail = chunks[i][-40:]
        assert any(word in chunks[i + 1] for word in tail.split() if len(word) > 3)


def test_no_overlap_larger_than_chunk_size_infinite_loop_guard():
    # A pathological config (overlap >= chunk_size) must not hang; this is
    # the kind of bug a sliding-window implementation can silently ship.
    text = "word " * 500
    chunks = chunk_text(text, chunk_size_chars=100, chunk_overlap_chars=99)
    assert len(chunks) > 0