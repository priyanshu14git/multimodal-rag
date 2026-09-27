"""
End-to-end multimodal RAG pipeline.

Architecture-style questions ("how does the model work", "explain the
encoder/decoder", etc.) get a boosted retrieval budget and a lightweight
rerank pass, because a single semantic nearest-neighbour search over
generic architecture prose tends to under-rank the one or two chunks/
diagrams that actually describe *this* document's model:

    Question
        ├──> text retrieval  ──> generic + question-term-aware reranking
        │
        └──> image retrieval ──> page-proximity + caption-aware selection
                         ↓
                    Vision LLM
                         ↓
                  grounded answer

Everything below is document-agnostic: nothing here is tuned to any one
paper, model name, or architecture. The only "knowledge" this module has
about the current document comes from (a) the question text itself and
(b) generic technical/document vocabulary (encoder, decoder, stage, layer,
etc.) that would apply to *any* technical PDF.
"""

import re

from app.services.embeddings import (
    get_image_embedder,
    get_text_embedder,
)
from app.services.generation import generate_multimodal_answer
from app.services import vectorstore


# ============================================================
# Question classification
# ============================================================

_ARCHITECTURE_TERMS = [
    "architecture",
    "architectural",
    "architecture diagram",
    "model structure",
    "network structure",
    "model design",
    "network design",
    "diagram",
    "encoder",
    "decoder",
    "feature extraction",
    "feature propagation",
    "information flow",
    "how does the model work",
    "how does the network work",
    "proposed model",
    "proposed architecture",
    "modules",
    "stages",
    "upsampling",
    "downsampling",
    "pipeline",
    "block diagram",
    "system design",
]


def _is_architecture_question(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in _ARCHITECTURE_TERMS)


# ============================================================
# Tokenization
# ============================================================

STOP_WORDS = {
    "what", "does", "the", "is", "are", "was", "were",
    "this", "that", "these", "those", "and", "or", "of",
    "to", "in", "on", "for", "with", "a", "an", "how",
    "why", "where", "when", "which", "who", "describe",
    "explain", "look", "like", "about", "from", "into",
    "through", "can", "you", "me", "tell", "give", "show",
}


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {
        token
        for token in tokens
        if len(token) >= 3 and token not in STOP_WORDS
    }


# ============================================================
# Named / model-specific term extraction (fully generic)
# ============================================================
#
# Instead of hardcoding a list of model names that "count" as relevant
# (which only works for one paper), we pull candidate named-entity-like
# terms straight out of the *question*: acronyms, hyphenated/plus-joined
# names, and alphanumeric identifiers (e.g. "DFPIR++", "ResNet-50",
# "GPT-4", "U-Net", "YOLOv8"). Whatever the user asks about becomes the
# thing we boost for — this generalizes to any document.

_NAMED_TERM_PATTERN = re.compile(
    r"\b[A-Za-z]+(?:[+\-][A-Za-z0-9]+)*\+*\b"
)


def _looks_like_named_term(token: str) -> bool:
    """True for tokens that look like a proper model/module name rather
    than an ordinary English word: contains a digit, a +/-, or is a
    short all-caps acronym."""
    if any(ch.isdigit() for ch in token) or "+" in token or "-" in token:
        return True
    if token.isupper() and 2 <= len(token) <= 8:
        return True
    return False


def _extract_named_terms(question: str) -> set[str]:
    candidates = _NAMED_TERM_PATTERN.findall(question)
    return {c.lower() for c in candidates if _looks_like_named_term(c)}


# ============================================================
# Content quality detection (generic document structure, not
# tied to any subject matter)
# ============================================================

def _is_navigation_content(text: str) -> bool:
    normalized = " ".join(text.lower().split())

    markers = [
        "table of contents",
        "list of figures",
        "list of tables",
        "references",
        "bibliography",
        "chapter contents",
    ]

    if any(marker in normalized for marker in markers):
        return True

    section_numbers = re.findall(r"\b\d+(?:\.\d+){1,3}\b", text)
    if len(section_numbers) >= 4:
        return True

    dotted_leaders = re.findall(r"\.{4,}", text)
    if len(dotted_leaders) >= 2:
        return True

    return False


def _is_list_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    words = stripped.split()
    if len(words) < 12:
        return True

    sentence_count = len(re.findall(r"[.!?](?:\s|$)", stripped))
    if len(words) >= 25 and sentence_count == 0:
        return True

    return False


# ============================================================
# General technical relevance (generic vocabulary that applies
# to basically any technical/ML paper, not one specific model)
# ============================================================

_TECHNICAL_TERMS = [
    "architecture",
    "network",
    "model",
    "encoder",
    "decoder",
    "feature",
    "convolution",
    "attention",
    "transformer",
    "resblock",
    "residual",
    "training",
    "loss",
    "module",
    "block",
    "input",
    "output",
    "upsampling",
    "downsampling",
    "embedding",
    "pipeline",
    "layer",
    "dataset",
    "channel",
    "resolution",
    "kernel",
    "stage",
    "concatenation",
    "skip connection",
]


def _technical_score(text: str) -> float:
    normalized = text.lower()
    matches = sum(1 for term in _TECHNICAL_TERMS if term in normalized)
    return min(matches / 8.0, 1.0)


# ============================================================
# Question-term relevance (replaces the old hardcoded,
# document-specific "_architecture_score")
# ============================================================

def _question_relevance_score(
    question: str,
    document: str,
    question_named_terms: set[str],
) -> float:
    """Generic relevance boost: how strongly does this chunk relate to
    *this* question, using only signals derived from the question itself
    plus generic technical vocabulary. Nothing here is specific to any
    one paper or model."""
    text = document.lower()
    score = 0.0

    # Exact named-term matches (e.g. the model name the user asked about).
    # This is the generalized replacement for hardcoded model-name lists.
    for term in question_named_terms:
        if term in text:
            score += 6.0

    # General lexical overlap between question and document.
    q_tokens = _tokenize(question)
    text_tokens = _tokenize(document)
    if q_tokens:
        overlap = len(q_tokens & text_tokens) / len(q_tokens)
        score += overlap * 5.0

    # Generic technical density (encoder/decoder/stage/etc.) helps
    # surface architecture-describing prose over incidental mentions.
    score += _technical_score(document) * 3.0

    # Navigation/reference material is never useful evidence.
    if _is_navigation_content(document):
        score -= 12.0

    return score


# ============================================================
# Text reranking
# ============================================================

def _rerank_text_results(
    question: str,
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
    max_results: int,
):
    architecture_question = _is_architecture_question(question)
    question_tokens = _tokenize(question)
    question_named_terms = _extract_named_terms(question)

    candidates = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        document = document or ""
        document_tokens = _tokenize(document)

        if question_tokens:
            lexical_overlap = len(question_tokens & document_tokens) / len(question_tokens)
        else:
            lexical_overlap = 0.0

        technical = _technical_score(document)

        # Chroma distance: lower is better.
        score = float(distance)
        score -= 0.15 * lexical_overlap
        score -= 0.08 * technical

        if _is_navigation_content(document):
            score += 0.80
        elif _is_list_like(document):
            score += 0.20

        # For architecture-style questions, weight in the generic
        # question-relevance score (named terms + overlap + technical
        # density), scaled so it meaningfully moves the ranking.
        if architecture_question:
            relevance = _question_relevance_score(
                question, document, question_named_terms
            )
            score -= 0.10 * relevance

        candidates.append(
            {
                "score": score,
                "document": document,
                "metadata": metadata,
                "distance": distance,
            }
        )

    candidates.sort(key=lambda item: item["score"])

    selected = []
    pages_seen = set()

    # Prefer different pages so we don't return several near-duplicate
    # chunks from the same page.
    for candidate in candidates:
        page = candidate["metadata"].get("page")
        if page not in pages_seen:
            selected.append(candidate)
            pages_seen.add(page)
        if len(selected) >= max_results:
            break

    if len(selected) < max_results:
        selected_ids = {id(c) for c in selected}
        for candidate in candidates:
            if id(candidate) in selected_ids:
                continue
            selected.append(candidate)
            if len(selected) >= max_results:
                break

    return (
        [item["document"] for item in selected],
        [item["metadata"] for item in selected],
        [item["distance"] for item in selected],
    )


# ============================================================
# Architecture image selection
# ============================================================

_DIAGRAM_CAPTION_TERMS = [
    "architecture",
    "framework",
    "network",
    "model",
    "encoder",
    "decoder",
    "module",
    "pipeline",
    "overview",
    "block diagram",
]


def _select_architecture_images(
    question: str,
    image_metadatas: list[dict],
    image_distances: list[float],
    selected_text_metadatas: list[dict],
):
    """Select the image most likely to be the architecture diagram.

    For architecture questions, image semantic similarity alone isn't
    enough (a lot of figures look visually similar to an embedding
    model). We combine it with page proximity to the strongest
    architecture-related text pages and generic caption vocabulary —
    none of which is specific to any one document.
    """
    if not image_metadatas:
        return [], []

    architecture_pages = [
        m.get("page") for m in selected_text_metadatas if m.get("page") is not None
    ]

    candidates = []

    for metadata, distance in zip(image_metadatas, image_distances):
        page = metadata.get("page")
        score = float(distance)

        if page is not None and architecture_pages:
            nearest_distance = min(abs(page - p) for p in architecture_pages)
            if nearest_distance == 0:
                score -= 0.40
            elif nearest_distance == 1:
                score -= 0.25
            elif nearest_distance == 2:
                score -= 0.12

        caption = (metadata.get("caption") or "").lower()
        if any(term in caption for term in _DIAGRAM_CAPTION_TERMS):
            score -= 0.20

        candidates.append({"score": score, "metadata": metadata, "distance": distance})

    candidates.sort(key=lambda item: item["score"])
    selected = candidates[:1]

    return (
        [item["metadata"] for item in selected],
        [item["distance"] for item in selected],
    )


# ============================================================
# Main RAG pipeline
# ============================================================

def answer_multimodal_question(
    question: str,
    doc_id: str,
    n_text_results: int = 5,
    n_image_results: int = 3,
    include_text: bool = True,
    include_images: bool = True,
    include_tables: bool = True,
    temperature: float = 0.15,
) -> dict:
    client = vectorstore.get_client()
    text_embedder = get_text_embedder()
    image_embedder = get_image_embedder()

    architecture_question = _is_architecture_question(question)

    # ========================================================
    # TEXT RETRIEVAL
    # ========================================================
    if not include_text:
        documents, metadatas, distances = [], [], []
    else:
        if architecture_question:
            # Architecture-style questions get a wider candidate pool
            # because semantic retrieval alone often ranks generic
            # technical prose above the specific passage that describes
            # this document's model.
            candidate_text_results = max(n_text_results * 8, 40)
        else:
            candidate_text_results = max(n_text_results * 3, 12)

        text_results = vectorstore.query_text(
            question=question,
            client=client,
            text_embedder=text_embedder,
            doc_id=doc_id,
            n_results=candidate_text_results,
        )

        documents = text_results["documents"][0]
        metadatas = text_results["metadatas"][0]
        distances = text_results["distances"][0]

        if not include_tables:
            # Tables and prose text share one collection (both are
            # embedded the same way), so "disable tables" is applied as
            # a pre-filter on the candidate pool rather than a separate
            # retrieval call.
            filtered = [
                (d, m, dist)
                for d, m, dist in zip(documents, metadatas, distances)
                if m.get("type") != "table"
            ]
            documents = [d for d, _, _ in filtered]
            metadatas = [m for _, m, _ in filtered]
            distances = [dist for _, _, dist in filtered]

        documents, metadatas, distances = _rerank_text_results(
            question=question,
            documents=documents,
            metadatas=metadatas,
            distances=distances,
            max_results=n_text_results,
        )

    # ========================================================
    # TEXT CONTEXT
    # ========================================================
    context_parts = []
    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        source_id = f"S{i}"
        document = document[:1200]
        context_parts.append(
            f"[{source_id}]\nPage: {metadata.get('page')}\nType: {metadata.get('type')}\nContent:\n{document}\n"
        )
    context = "\n\n".join(context_parts)

    # ========================================================
    # IMAGE RETRIEVAL
    # ========================================================
    if not include_images:
        image_metadatas, image_distances = [], []
    else:
        if architecture_question:
            image_candidate_count = max(n_image_results * 4, 10)
        else:
            image_candidate_count = n_image_results

        image_query_embedding = image_embedder.encode_text(question)

        image_results = vectorstore.query_image(
            image_embedding=image_query_embedding,
            client=client,
            doc_id=doc_id,
            n_results=image_candidate_count,
        )

        all_image_metadatas = image_results["metadatas"][0]
        all_image_distances = image_results["distances"][0]

        if architecture_question:
            image_metadatas, image_distances = _select_architecture_images(
                question=question,
                image_metadatas=all_image_metadatas,
                image_distances=all_image_distances,
                selected_text_metadatas=metadatas,
            )
        else:
            image_metadatas = all_image_metadatas
            image_distances = all_image_distances

    # ========================================================
    # IMAGE CONTEXT
    # ========================================================
    image_paths = []
    image_context_parts = []
    for i, metadata in enumerate(image_metadatas, start=1):
        image_id = f"I{i}"
        image_path = metadata.get("file_path")
        if image_path:
            image_paths.append(image_path)
        image_context_parts.append(
            f"[{image_id}]\nImage source: {image_id}\nPage: {metadata.get('page')}\n"
            f"Type: {metadata.get('type')}\nCaption: {metadata.get('caption') or 'No caption available'}\n"
        )
    image_context = "\n\n".join(image_context_parts)

    # ========================================================
    # GENERATION
    # ========================================================
    answer = generate_multimodal_answer(
        question=question,
        context=context,
        image_paths=image_paths,
        image_context=image_context,
        temperature=temperature,
    )

    # A minimal, honest grounding signal: true only if we actually found
    # some evidence to answer from. This doesn't inspect the generated
    # answer text (that would risk false confidence) - it just reflects
    # whether retrieval returned anything at all.
    grounded = bool(documents) or bool(image_metadatas)

    # ========================================================
    # CITATIONS
    # ========================================================
    citations = []
    for i, metadata in enumerate(metadatas, start=1):
        citations.append(
            {
                "source_id": f"S{i}",
                "page": metadata.get("page"),
                "type": metadata.get("type"),
                "label": f"{metadata.get('type', 'text').capitalize()} source",
            }
        )
    for i, metadata in enumerate(image_metadatas, start=1):
        citations.append(
            {
                "source_id": f"I{i}",
                "page": metadata.get("page"),
                "type": metadata.get("type"),
                "label": "Image source",
            }
        )

    # ========================================================
    # RETURN
    # ========================================================
    return {
        "answer": answer,
        "grounded": grounded,
        "citations": citations,
        "retrieved_text": [
            {
                "source_id": f"S{i}",
                "page": metadata.get("page"),
                "type": metadata.get("type"),
                "distance": distance,
                "content": document,
            }
            for i, (document, metadata, distance) in enumerate(
                zip(documents, metadatas, distances), start=1
            )
        ],
        "retrieved_images": [
            {
                "source_id": f"I{i}",
                "page": metadata.get("page"),
                "type": metadata.get("type"),
                "distance": distance,
                "file_path": metadata.get("file_path"),
                "caption": metadata.get("caption"),
            }
            for i, (metadata, distance) in enumerate(
                zip(image_metadatas, image_distances), start=1
            )
        ],
    }