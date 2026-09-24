"""
Simple, dependency-free text chunker.

We deliberately don't pull in a heavy splitter library for this: a
paragraph/sentence-aware sliding window is easy to reason about, easy to
explain in an interview, and good enough quality-wise for a resume project.
If eval later shows retrieval quality suffers on very long dense pages,
swapping this for `langchain_text_splitters.RecursiveCharacterTextSplitter`
is a one-function change (documented in README "what I'd do next").
"""
import re


def _split_into_sentences(text: str) -> list[str]:
    # Good-enough sentence boundary heuristic (period/question/exclaim +
    # whitespace + capital letter). Not linguistically perfect, but avoids
    # pulling in nltk/spacy for a resume project.
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size_chars: int = 1000,
    chunk_overlap_chars: int = 150,
) -> list[str]:
    """Greedily pack sentences into chunks up to `chunk_size_chars`, then
    carry the last `chunk_overlap_chars` worth of sentences into the next
    chunk so context isn't lost across a chunk boundary."""
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size_chars and current:
            chunks.append(" ".join(current))
            # carry overlap: keep trailing sentences whose combined length
            # is <= chunk_overlap_chars
            overlap: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > chunk_overlap_chars:
                    break
                overlap.insert(0, s)
                overlap_len += len(s)
            current = overlap
            current_len = overlap_len

        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks
