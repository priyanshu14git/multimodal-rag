"""
Shared data shapes.

Keeping these as pydantic models (not raw dicts) gives us validation for
free and means the ingestion pipeline, vector store, and API layer are all
speaking the same language. `bbox` is (x0, y0, x1, y1) in PDF point units,
top-left origin, matching PyMuPDF's convention.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class TextChunk(BaseModel):
    id: str
    doc_id: str
    type: ElementType = ElementType.TEXT
    page: int
    text: str
    bbox: Optional[BBox] = None
    chunk_index: int  # position among chunks on this page, for citation text


class TableElement(BaseModel):
    id: str
    doc_id: str
    type: ElementType = ElementType.TABLE
    page: int
    # Table content flattened to markdown so it can be embedded and shown
    # to the LLM like any other text block, while `rows` keeps the
    # structured form around for the UI to render as an actual table.
    markdown: str
    rows: list[list[Optional[str]]]
    bbox: Optional[BBox] = None
    table_index: int  # nth table on this page


class ImageElement(BaseModel):
    id: str
    doc_id: str
    type: ElementType = ElementType.IMAGE
    page: int
    file_path: str  # relative path under backend/data/processed/
    bbox: Optional[BBox] = None
    image_index: int  # nth image on this page
    caption: Optional[str] = None  # nearest text block above/below, if found


class IngestManifest(BaseModel):
    """Everything extracted from one PDF. This is what gets written to
    backend/data/processed/<doc_id>/manifest.json and later fed into the
    embedding stage."""

    doc_id: str
    source_filename: str
    num_pages: int
    text_chunks: list[TextChunk] = Field(default_factory=list)
    tables: list[TableElement] = Field(default_factory=list)
    images: list[ImageElement] = Field(default_factory=list)


class IngestResponse(BaseModel):
    doc_id: str
    num_pages: int
    num_text_chunks: int
    num_tables: int
    num_images: int


class RetrievedElement(BaseModel):
    """A single piece of retrieved context, normalized across text/table/image
    so the generation and frontend layers don't need to special-case type."""

    element: TextChunk | TableElement | ImageElement
    score: float


class AskRequest(BaseModel):
    doc_id: str
    question: str
    # --- Optional retrieval/generation controls (all default to the
    # pipeline's original fixed behavior, so existing callers that omit
    # these fields see no change) -------------------------------------
    top_k_text: int = 5
    top_k_images: int = 3
    include_text: bool = True
    include_images: bool = True
    include_tables: bool = True
    temperature: float = 0.15


class Citation(BaseModel):
    page: int
    type: ElementType
    label: str  # e.g. "page 4, Figure 2" or "page 2, Table 1"


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedElement]
    grounded: bool  # False if the guardrail triggered (no relevant content found)