"""
Pure helper functions used by the Streamlit frontend, kept in their own
module (rather than inline in app.py) so they're importable in tests
without pulling in Streamlit's script-execution machinery.
"""
import re

import pandas as pd

_SEPARATOR_ROW = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


def parse_markdown_table(md: str):
    """Parse the pipe-delimited markdown tables the backend emits (see
    backend/app/services/pdf_parser.py::_rows_to_markdown) into a
    DataFrame, so retrieved tables render as real tables instead of raw
    markup. Returns None if the content doesn't look like a table -
    callers fall back to showing the raw markdown in that case."""
    if not md or "|" not in md:
        return None

    rows = []
    for line in md.strip().splitlines():
        line = line.strip()
        if not line or _SEPARATOR_ROW.match(line):
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])

    if len(rows) < 1:
        return None

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    try:
        header, *body = rows
        if not body:
            return None
        return pd.DataFrame(body, columns=header)
    except Exception:
        return None