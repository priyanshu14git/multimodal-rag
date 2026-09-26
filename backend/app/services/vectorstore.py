"""
ChromaDB vector store for the multimodal RAG pipeline.

We keep two collections:
    1. text_elements  -> text chunks + tables
    2. image_elements -> extracted PDF images

Text/tables use BGE embeddings (384 dimensions).
Images use SigLIP embeddings (768 dimensions).
"""

from pathlib import Path
from typing import Any

import chromadb

from app.models.schemas import (
    ElementType,
    ImageElement,
    IngestManifest,
    TableElement,
    TextChunk,
)


# Store Chroma data inside the project.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHROMA_DIR = PROJECT_ROOT / "backend" / "data" / "chroma"


def get_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_text_collection(client: chromadb.PersistentClient):
    """Collection for text chunks and tables."""
    return client.get_or_create_collection(
        name="text_elements",
        metadata={"hnsw:space": "cosine"},
    )


def get_image_collection(client: chromadb.PersistentClient):
    """Collection for image embeddings."""
    return client.get_or_create_collection(
        name="image_elements",
        metadata={"hnsw:space": "cosine"},
    )


def _text_element_content(element: TextChunk | TableElement) -> str:
    """Return the text that should be embedded."""
    if isinstance(element, TextChunk):
        return element.text

    return element.markdown


def _text_metadata(element: TextChunk | TableElement) -> dict[str, Any]:
    """Metadata stored alongside a text/table embedding."""
    metadata: dict[str, Any] = {
        "doc_id": element.doc_id,
        "page": element.page,
        "type": element.type.value,
    }

    if element.bbox is not None:
        metadata.update(
            {
                "bbox_x0": element.bbox.x0,
                "bbox_y0": element.bbox.y0,
                "bbox_x1": element.bbox.x1,
                "bbox_y1": element.bbox.y1,
            }
        )

    if isinstance(element, TextChunk):
        metadata["chunk_index"] = element.chunk_index
    else:
        metadata["table_index"] = element.table_index

    return metadata


def _image_metadata(element: ImageElement) -> dict[str, Any]:
    """Metadata stored alongside an image embedding."""
    metadata: dict[str, Any] = {
        "doc_id": element.doc_id,
        "page": element.page,
        "type": element.type.value,
        "file_path": element.file_path,
        "image_index": element.image_index,
    }

    if element.caption:
        metadata["caption"] = element.caption

    if element.bbox is not None:
        metadata.update(
            {
                "bbox_x0": element.bbox.x0,
                "bbox_y0": element.bbox.y0,
                "bbox_x1": element.bbox.x1,
                "bbox_y1": element.bbox.y1,
            }
        )

    return metadata


def index_manifest(
    manifest: IngestManifest,
    client: chromadb.PersistentClient,
    text_embedder,
    image_embedder,
) -> dict[str, int]:
    """
    Embed and index everything from one ingestion manifest.

    Returns counts of indexed text/table/image elements.
    """
    text_collection = get_text_collection(client)
    image_collection = get_image_collection(client)

    text_ids: list[str] = []
    text_documents: list[str] = []
    text_embeddings: list[list[float]] = []
    text_metadatas: list[dict[str, Any]] = []

    # ---------------------------------------------------------------
    # Text chunks
    # ---------------------------------------------------------------
    for element in manifest.text_chunks:
        content = _text_element_content(element)

        text_ids.append(element.id)
        text_documents.append(content)
        embedding = text_embedder.encode(content)

        if hasattr(embedding, "tolist"):
          embedding = embedding.tolist()

        text_embeddings.append(embedding)
        text_metadatas.append(_text_metadata(element))

    # ---------------------------------------------------------------
    # Tables
    # ---------------------------------------------------------------
    for element in manifest.tables:
        content = _text_element_content(element)

        text_ids.append(element.id)
        text_documents.append(content)
        embedding = text_embedder.encode(content)

        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        text_embeddings.append(embedding)
        text_metadatas.append(_text_metadata(element))

    if text_ids:
        text_collection.upsert(
            ids=text_ids,
            documents=text_documents,
            embeddings=text_embeddings,
            metadatas=text_metadatas,
        )

    # ---------------------------------------------------------------
    # Images
    # ---------------------------------------------------------------
    image_ids: list[str] = []
    image_embeddings: list[list[float]] = []
    image_metadatas: list[dict[str, Any]] = []

    for element in manifest.images:
        image_path = Path(element.file_path)

        embedding = image_embedder.encode_image(image_path)

        image_ids.append(element.id)
        image_embeddings.append(embedding)
        image_metadatas.append(_image_metadata(element))

    if image_ids:
        image_collection.upsert(
            ids=image_ids,
            embeddings=image_embeddings,
            metadatas=image_metadatas,
        )

    return {
        "text_chunks": len(manifest.text_chunks),
        "tables": len(manifest.tables),
        "images": len(manifest.images),
    }


def query_text(
    question: str,
    client: chromadb.PersistentClient,
    text_embedder,
    doc_id: str,
    n_results: int = 5,
) -> dict[str, Any]:
    """Search text chunks/tables using a BGE embedding."""
    collection = get_text_collection(client)

    query_embedding = text_embedder.encode(question)

    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"doc_id": doc_id},
    )


def query_image(
    image_embedding: list[float],
    client: chromadb.PersistentClient,
    doc_id: str,
    n_results: int = 5,
) -> dict[str, Any]:
    """Search extracted images using a SigLIP embedding."""
    collection = get_image_collection(client)

    return collection.query(
        query_embeddings=[image_embedding],
        n_results=n_results,
        where={"doc_id": doc_id},
    )