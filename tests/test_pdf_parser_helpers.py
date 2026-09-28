"""
Unit tests for the pure helper functions in app.services.pdf_parser.

These only need PyMuPDF/pdfplumber importable (both are core, unavoidable
requirements.txt dependencies for this project) - they don't touch any
actual PDF file, so no fixtures are needed.
"""

from app.services.pdf_parser import _bbox_overlaps, _make_doc_id, _rows_to_markdown


# ---------------------------------------------------------------------------
# _bbox_overlaps
# ---------------------------------------------------------------------------

def test_bbox_overlaps_when_center_inside():
    text_block = (10, 10, 20, 20)  # center = (15, 15)
    table_bbox = (0, 0, 100, 100)
    assert _bbox_overlaps(text_block, table_bbox)


def test_bbox_does_not_overlap_when_center_outside():
    text_block = (200, 200, 210, 210)  # center = (205, 205)
    table_bbox = (0, 0, 100, 100)
    assert not _bbox_overlaps(text_block, table_bbox)


def test_bbox_overlap_is_center_based_not_full_containment():
    # A block that's mostly outside the table but whose center still falls
    # inside should count as overlapping - this documents the deliberate
    # "center point" heuristic rather than requiring full containment.
    text_block = (90, 90, 150, 150)  # center = (120, 120)
    table_bbox = (0, 0, 130, 130)
    assert _bbox_overlaps(text_block, table_bbox)


# ---------------------------------------------------------------------------
# _rows_to_markdown
# ---------------------------------------------------------------------------

def test_rows_to_markdown_basic_table():
    rows = [["Quarter", "Revenue"], ["Q1", "12"], ["Q2", "19"]]
    md = _rows_to_markdown(rows)
    lines = md.splitlines()
    assert lines[0] == "| Quarter | Revenue |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| Q1 | 12 |"
    assert lines[3] == "| Q2 | 19 |"


def test_rows_to_markdown_handles_none_cells():
    rows = [["A", "B"], [None, "x"], ["y", None]]
    md = _rows_to_markdown(rows)
    assert "|  | x |" in md
    assert "| y |  |" in md


def test_rows_to_markdown_strips_newlines_in_cells():
    rows = [["A"], ["multi\nline\ncell"]]
    md = _rows_to_markdown(rows)
    assert "\n" not in md.splitlines()[-1]
    assert "multi line cell" in md.splitlines()[-1]


def test_rows_to_markdown_empty_input():
    assert _rows_to_markdown([]) == ""


# ---------------------------------------------------------------------------
# _make_doc_id
# ---------------------------------------------------------------------------

def test_make_doc_id_is_deterministic_for_same_content(tmp_path):
    pdf1 = tmp_path / "report.pdf"
    pdf1.write_bytes(b"%PDF-1.4 fake content for hashing")
    id_a = _make_doc_id(pdf1)
    id_b = _make_doc_id(pdf1)
    assert id_a == id_b


def test_make_doc_id_differs_for_different_content(tmp_path):
    pdf_a = tmp_path / "a.pdf"
    pdf_a.write_bytes(b"content A")
    pdf_b = tmp_path / "b.pdf"
    pdf_b.write_bytes(b"content B")
    assert _make_doc_id(pdf_a) != _make_doc_id(pdf_b)


def test_make_doc_id_sanitizes_unsafe_filename_characters(tmp_path):
    pdf = tmp_path / "my report (final v2)!.pdf"
    pdf.write_bytes(b"some bytes")
    doc_id = _make_doc_id(pdf)
    assert " " not in doc_id
    assert "(" not in doc_id and ")" not in doc_id
    assert "!" not in doc_id