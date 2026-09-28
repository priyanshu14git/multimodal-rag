"""
Unit tests for frontend/utils.py::parse_markdown_table.

Deliberately does NOT import frontend/app.py (which calls st.set_page_config
and main() at import time) - see utils.py's own docstring for why this
logic lives in its own module.
"""

import pandas as pd

from utils import parse_markdown_table


def test_parses_simple_table():
    md = (
        "| Quarter | Revenue ($M) | Growth (%) |\n"
        "| --- | --- | --- |\n"
        "| Q1 | 12 | — |\n"
        "| Q2 | 19 | 58% |\n"
    )
    df = parse_markdown_table(md)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Quarter", "Revenue ($M)", "Growth (%)"]
    assert len(df) == 2
    assert df.iloc[0]["Quarter"] == "Q1"
    assert df.iloc[1]["Growth (%)"] == "58%"


def test_returns_none_for_non_table_text():
    assert parse_markdown_table("This is just a plain sentence, no pipes here.") is None
    assert parse_markdown_table("") is None
    assert parse_markdown_table(None) is None


def test_returns_none_for_header_only_table():
    md = "| A | B |\n| --- | --- |\n"
    assert parse_markdown_table(md) is None


def test_handles_ragged_rows_without_crashing():
    # A row with fewer cells than the header shouldn't raise - it should
    # get padded, not crash the whole render.
    md = (
        "| A | B | C |\n"
        "| --- | --- | --- |\n"
        "| 1 | 2 |\n"
    )
    df = parse_markdown_table(md)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["A", "B", "C"]
    assert df.iloc[0]["C"] == ""


def test_separator_row_variants_are_ignored():
    md = (
        "| A | B |\n"
        "|:---|---:|\n"
        "| x | y |\n"
    )
    df = parse_markdown_table(md)
    assert df is not None
    assert len(df) == 1