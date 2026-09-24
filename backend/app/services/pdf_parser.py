"""
PDF -> IngestManifest.

Design notes (useful for interviews):

- Text extraction is done at PyMuPDF *block* granularity rather than whole-page
  text, because blocks come with real bounding boxes for free
  (`page.get_text("blocks")`). A block that's still bigger than
  `chunk_size_chars` gets sub-chunked with `chunking.chunk_text`, and all
  sub-chunks inherit the parent block's bbox. This is an approximation
  (a sub-chunk technically only occupies part of the block) but keeps
  bbox provenance both real and simple. Documented as a known limitation.
- Tables use pdfplumber instead of PyMuPDF, because PyMuPDF has no table
  structure detection and pdfplumber's `find_tables()` gives both cell
  structure *and* a bounding box.
- Images are extracted via PyMuPDF's `get_images` + `extract_image`, saved
  as PNG/JPEG files under `data/processed/<doc_id>/images/`, and matched to
  a bbox via `get_image_rects`. A lightweight caption heuristic looks for
  the nearest text block directly below the image.
"""
import hashlib
import json
import re
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from app.models.schemas import BBox, ImageElement, IngestManifest, TableElement, TextChunk
from app.services.chunking import chunk_text


def _make_doc_id(pdf_path: Path) -> str:
    digest = hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:8]
    stem = re.sub(r"[^a-zA-Z0-9_-]", "_", pdf_path.stem)
    return f"{stem}_{digest}"


def _bbox_overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """True if bbox `a`'s center falls inside bbox `b` (used to drop text
    blocks that PyMuPDF also extracted from inside a detected table, so the
    same cell text isn't indexed twice - once garbled, once as clean
    markdown)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    cx, cy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
    return bx0 <= cx <= bx1 and by0 <= cy <= by1


def _extract_text_chunks(
    page: "fitz.Page",
    page_num: int,
    doc_id: str,
    chunk_size: int,
    overlap: int,
    table_bboxes: list[tuple[float, float, float, float]],
) -> list[TextChunk]:
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    # keep text blocks only (block_type 0), reading order top-to-bottom, left-to-right,
    # and drop any block that sits inside an already-detected table (see
    # `_bbox_overlaps` docstring).
    text_blocks = sorted(
        (
            b
            for b in blocks
            if b[6] == 0
            and b[4].strip()
            and not any(_bbox_overlaps(b[:4], tb) for tb in table_bboxes)
        ),
        key=lambda b: (round(b[1], 1), b[0]),
    )

    chunks: list[TextChunk] = []
    chunk_index = 0
    for x0, y0, x1, y1, text, *_ in text_blocks:
        text = text.strip()
        if not text:
            continue
        bbox = BBox(x0=x0, y0=y0, x1=x1, y1=y1)
        if len(text) <= int(chunk_size * 1.5):
            pieces = [text]
        else:
            pieces = chunk_text(text, chunk_size, overlap)
        for piece in pieces:
            chunks.append(
                TextChunk(
                    id=f"{doc_id}_p{page_num}_t{chunk_index}",
                    doc_id=doc_id,
                    page=page_num,
                    text=piece,
                    bbox=bbox,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
    return chunks


def _rows_to_markdown(rows: list[list[str | None]]) -> str:
    if not rows:
        return ""
    header, *body = rows
    header_cells = [c or "" for c in header]
    lines = ["| " + " | ".join(header_cells) + " |", "| " + " | ".join(["---"] * len(header_cells)) + " |"]
    for row in body:
        cells = [(c or "").replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _extract_tables(pdf_path: Path, page_num: int, doc_id: str) -> list[TableElement]:
    tables: list[TableElement] = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        for idx, table in enumerate(page.find_tables()):
            rows = table.extract()
            if not rows or all(not any(cell for cell in row) for row in rows):
                continue
            x0, y0, x1, y1 = table.bbox
            tables.append(
                TableElement(
                    id=f"{doc_id}_p{page_num}_tbl{idx}",
                    doc_id=doc_id,
                    page=page_num,
                    markdown=_rows_to_markdown(rows),
                    rows=rows,
                    bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                    table_index=idx,
                )
            )
    return tables


_CAPTION_PATTERN = re.compile(r"^(figure|fig\.?|table|chart)\s*\d+", re.IGNORECASE)


def _find_caption(page: "fitz.Page", image_bbox: tuple[float, float, float, float]) -> str | None:
    """Heuristic: look at text blocks within 40pt directly above or below the
    image. Prefer one that starts with "Figure N" / "Table N" / etc. (a real
    caption almost always does); otherwise fall back to the shortest nearby
    block, since a caption is usually much shorter than a body paragraph."""
    img_x0, img_y0, img_x1, img_y1 = image_bbox
    candidates = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        text = text.strip()
        if not text:
            continue
        # allow a small negative gap too: tight layouts can have the caption
        # block's bbox touch or slightly overlap the image's bbox
        near_below = -5 <= (y0 - img_y1) <= 40
        near_above = -5 <= (img_y0 - y1) <= 40
        if near_below or near_above:
            candidates.append(text)

    if not candidates:
        return None

    labeled = [c for c in candidates if _CAPTION_PATTERN.match(c)]
    pool = labeled or candidates
    return min(pool, key=len)[:200]


def _extract_images(
    doc: "fitz.Document", page: "fitz.Page", page_num: int, doc_id: str, images_dir: Path
) -> list[ImageElement]:
    elements: list[ImageElement] = []
    seen_xrefs: set[int] = set()

    for idx, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        rects = page.get_image_rects(xref)
        bbox = None
        if rects:
            r = rects[0]
            bbox = BBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)

        try:
            base_image = doc.extract_image(xref)
        except Exception:
            continue
        ext = base_image.get("ext", "png")
        image_bytes = base_image["image"]

        # Skip tiny images (icons, bullets, decorative artifacts) - not
        # useful retrieval targets and just add noise to the image index.
        if base_image.get("width", 0) < 50 or base_image.get("height", 0) < 50:
            continue

        filename = f"p{page_num}_img{idx}.{ext}"
        file_path = images_dir / filename
        file_path.write_bytes(image_bytes)

        caption = None
        if rects:
            caption = _find_caption(page, (r.x0, r.y0, r.x1, r.y1))

        elements.append(
            ImageElement(
                id=f"{doc_id}_p{page_num}_img{idx}",
                doc_id=doc_id,
                page=page_num,
                file_path=str(file_path),
                bbox=bbox,
                image_index=idx,
                caption=caption,
            )
        )
    return elements


def parse_pdf(
    pdf_path: str | Path,
    processed_dir: str | Path,
    chunk_size_chars: int = 1000,
    chunk_overlap_chars: int = 150,
) -> IngestManifest:
    """Parse one PDF into a full IngestManifest and persist it (plus
    extracted images) under `processed_dir/<doc_id>/`."""
    pdf_path = Path(pdf_path)
    processed_dir = Path(processed_dir)
    doc_id = _make_doc_id(pdf_path)

    doc_out_dir = processed_dir / doc_id
    images_dir = doc_out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    manifest = IngestManifest(
        doc_id=doc_id, source_filename=pdf_path.name, num_pages=doc.page_count
    )

    for page_index in range(doc.page_count):
        page_num = page_index + 1  # 1-indexed for human-friendly citations
        page = doc[page_index]

        page_tables = _extract_tables(pdf_path, page_num, doc_id)
        table_bboxes = [(t.bbox.x0, t.bbox.y0, t.bbox.x1, t.bbox.y1) for t in page_tables if t.bbox]

        manifest.text_chunks.extend(
            _extract_text_chunks(
                page, page_num, doc_id, chunk_size_chars, chunk_overlap_chars, table_bboxes
            )
        )
        manifest.tables.extend(page_tables)
        manifest.images.extend(_extract_images(doc, page, page_num, doc_id, images_dir))

    doc.close()

    manifest_path = doc_out_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))

    return manifest
