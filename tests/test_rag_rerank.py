"""
Unit tests for the pure, document-agnostic classification/reranking
functions in app.services.rag.

These test the generalized reranker (see README "Design decisions" and
the git history around commit "generalize retrieval reranking") — the
whole point of these tests is to catch a regression back toward hardcoding
scores for specific model names, so several tests below deliberately use
made-up model names that don't appear in any real document.
"""

from app.services.rag import (
    _extract_named_terms,
    _is_architecture_question,
    _is_list_like,
    _is_navigation_content,
    _rerank_text_results,
    _technical_score,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Question classification
# ---------------------------------------------------------------------------

def test_architecture_question_detected():
    assert _is_architecture_question("Explain the architecture shown in the document.")
    assert _is_architecture_question("How does the encoder work?")
    assert _is_architecture_question("Walk me through the model's pipeline.")


def test_non_architecture_question_not_flagged():
    assert not _is_architecture_question("What does Table 3 show?")
    assert not _is_architecture_question("Summarize the main findings.")


# ---------------------------------------------------------------------------
# Named-term extraction: this is the generalized replacement for the old
# hardcoded model-name list, so it should work for ANY made-up model name.
# ---------------------------------------------------------------------------

def test_extracts_hyphenated_and_plus_named_terms():
    terms = _extract_named_terms("Explain the ZorbNet-XL++ architecture.")
    assert "zorbnet-xl++" in terms


def test_extracts_acronym_and_versioned_names():
    terms = _extract_named_terms("How does ResNet-50 compare to GPT-4?")
    assert "resnet-50" in terms
    assert "gpt-4" in terms


def test_does_not_extract_ordinary_words_as_named_terms():
    terms = _extract_named_terms("Explain the encoder and decoder stages.")
    assert "encoder" not in terms
    assert "decoder" not in terms
    assert "stages" not in terms


# ---------------------------------------------------------------------------
# Tokenization / technical scoring
# ---------------------------------------------------------------------------

def test_tokenize_drops_stop_words_and_short_tokens():
    tokens = _tokenize("What is the encoder doing in this model?")
    assert "encoder" in tokens
    assert "model" in tokens
    assert "what" not in tokens
    assert "is" not in tokens


def test_technical_score_higher_for_technical_text():
    technical = "The encoder uses residual convolution blocks with attention and skip connections."
    plain = "The weather was nice and the team went for a walk in the park."
    assert _technical_score(technical) > _technical_score(plain)


# ---------------------------------------------------------------------------
# Content quality filters
# ---------------------------------------------------------------------------

def test_table_of_contents_flagged_as_navigation():
    toc = "Table of Contents\n1.1 Introduction .......... 3\n1.2 Background .......... 5"
    assert _is_navigation_content(toc)


def test_normal_prose_not_flagged_as_navigation():
    prose = (
        "The encoder progressively downsamples the input feature map while "
        "increasing channel depth, before passing the bottleneck representation "
        "to the decoder for upsampling."
    )
    assert not _is_navigation_content(prose)


def test_short_fragment_is_list_like():
    assert _is_list_like("Figure 3.")
    assert _is_list_like("")


def test_full_sentence_is_not_list_like():
    sentence = (
        "The decoder reconstructs the output image by progressively upsampling "
        "the bottleneck features and merging them with encoder skip connections."
    )
    assert not _is_list_like(sentence)


# ---------------------------------------------------------------------------
# End-to-end reranking behavior, with a document invented for this test -
# proves the reranker generalizes rather than depending on any real paper.
# ---------------------------------------------------------------------------

def _doc(text, page, type_="text"):
    return text, {"page": page, "type": type_}


def test_reranker_prefers_named_term_match_for_architecture_question():
    question = "Explain the ZorbNet-XL++ architecture."

    documents = []
    metadatas = []
    distances = []

    # Chunk 1: navigation noise, should rank last regardless of distance.
    d, m = _doc("Table of Contents\n1.1 .......... 3\n1.2 .......... 5", page=1)
    documents.append(d); metadatas.append(m); distances.append(0.05)

    # Chunk 2: generic technical prose that does NOT mention the asked-about
    # model - this is the "unrelated architecture" case, handled generically
    # now instead of via a hardcoded exclusion list.
    d, m = _doc(
        "The baseline WobbleNet encoder uses residual blocks and an attention "
        "module before downsampling into the bottleneck representation.",
        page=10,
    )
    documents.append(d); metadatas.append(m); distances.append(0.10)

    # Chunk 3: actually describes the architecture asked about, but with a
    # WORSE raw embedding distance than chunk 2 - this is the exact failure
    # mode the reranker exists to fix.
    d, m = _doc(
        "ZorbNet-XL++ is a lightweight encoder-decoder model. Its encoder "
        "downsamples the input before the decoder upsamples back to full "
        "resolution using skip connections.",
        page=25,
    )
    documents.append(d); metadatas.append(m); distances.append(0.30)

    reranked_docs, reranked_meta, _ = _rerank_text_results(
        question=question,
        documents=documents,
        metadatas=metadatas,
        distances=distances,
        max_results=3,
    )

    # The chunk actually describing the asked-about model should outrank
    # the unrelated-but-closer-in-embedding-space baseline chunk.
    assert reranked_meta[0]["page"] == 25
    # Navigation noise should never be ranked first.
    assert reranked_meta[0]["page"] != 1


def test_reranker_deduplicates_by_page_when_possible():
    question = "Explain the model architecture."
    documents, metadatas, distances = [], [], []

    for i in range(4):
        d, m = _doc(f"Architecture description chunk {i} about the model's encoder.", page=5)
        documents.append(d); metadatas.append(m); distances.append(0.1 + i * 0.01)

    d, m = _doc("A different architecture section describing the decoder.", page=6)
    documents.append(d); metadatas.append(m); distances.append(0.5)

    _, reranked_meta, _ = _rerank_text_results(
        question=question,
        documents=documents,
        metadatas=metadatas,
        distances=distances,
        max_results=2,
    )

    pages = [m["page"] for m in reranked_meta]
    # Should prefer covering two different pages over four chunks of the
    # same page, when max_results allows it.
    assert len(set(pages)) == len(pages)