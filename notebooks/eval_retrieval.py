"""
Small retrieval/answer-quality eval for the multimodal RAG pipeline.

This measures two things per question, against `notebooks/eval_questions.json`:

1. Retrieval relevance: did any returned citation land on (or near) a page
   we know contains the answer?
2. Answer faithfulness (proxy): what fraction of the expected keywords for
   that question actually show up in the generated answer? This is a rough
   proxy, not a real faithfulness metric (see README limitations) — it
   catches "the answer clearly didn't use the right evidence" cases, not
   subtle hallucination within an otherwise-correct answer.

It also checks the `grounded` flag against `expect_grounded` for the
"grounding" category questions (things the document plausibly can't
answer) — see the docstring on `check_grounding` for an important caveat
this eval is likely to surface.

Usage
-----
Against already-ingested documents (fast iteration):

    python notebooks/eval_retrieval.py \\
        --doc-map '{"surajreport": "surajreport_eeac2708", "sample_report": "sample_report_15661a62"}'

Or ingest fresh copies as part of the run (slower, but reproducible from a
clean state):

    python notebooks/eval_retrieval.py \\
        --pdf surajreport=backend/data/raw_pdfs/surajreport.pdf \\
        --pdf sample_report=backend/data/raw_pdfs/sample_report.pdf

Either way, a markdown report is written to notebooks/eval_report.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

API_URL = "http://127.0.0.1:8000"
QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
REPORT_PATH = Path(__file__).parent / "eval_report.md"


# =============================================================================
# Pure scoring functions (unit-tested in tests/test_eval_scoring.py)
# =============================================================================

def page_hit(citations: list[dict], expected_pages: list[int], tolerance: int = 0) -> bool:
    """True if any citation's page matches (within `tolerance` pages of)
    one of the expected pages. An empty expected_pages list means "no
    specific page to check" and always returns True - use expect_grounded
    for questions the document shouldn't be able to answer at all."""
    if not expected_pages:
        return True
    citation_pages = [c.get("page") for c in citations if c.get("page") is not None]
    for cp in citation_pages:
        for ep in expected_pages:
            if abs(cp - ep) <= tolerance:
                return True
    return False


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    """Fraction of `keywords` that appear (case-insensitively) in `answer`.
    Returns 1.0 for an empty keyword list (nothing to check)."""
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def check_grounding(actual_grounded: bool, expect_grounded: bool | None) -> bool | None:
    """Returns True/False if there's an expectation to check, None if this
    question didn't specify one (expect_grounded defaults to "don't care").

    Caveat this eval is likely to surface: the backend's `grounded` flag
    (see rag.py::answer_multimodal_question) is currently just "did
    retrieval return anything at all," not "was it actually relevant."
    Chroma always returns its nearest neighbors regardless of how distant
    they are, so a question the document can't answer may still come back
    grounded=True. config.py already defines `min_relevance_score` for
    exactly this purpose but nothing consults it yet — if this eval shows
    grounding checks failing on the "grounding" category questions, that's
    the reason, and wiring min_relevance_score into the grounded
    computation would be the fix.
    """
    if expect_grounded is None:
        return None
    return actual_grounded == expect_grounded


def evaluate_answer(question_spec: dict, response: dict, latency_s: float) -> dict:
    """Combine the pure checks above into one result row for a single
    question. `response` is the raw JSON body from POST /ask."""
    citations = response.get("citations", [])
    answer = response.get("answer", "")
    grounded = response.get("grounded", True)

    return {
        "id": question_spec["id"],
        "doc": question_spec["doc"],
        "category": question_spec["category"],
        "question": question_spec["question"],
        "page_hit": page_hit(citations, question_spec.get("expected_pages", []), tolerance=1),
        "keyword_coverage": keyword_coverage(answer, question_spec.get("expected_keywords", [])),
        "grounding_ok": check_grounding(grounded, question_spec.get("expect_grounded")),
        "grounded": grounded,
        "latency_s": round(latency_s, 2),
        "answer_preview": (answer[:180] + "…") if len(answer) > 180 else answer,
    }


# =============================================================================
# I/O: loading questions, ingesting, asking, reporting
# =============================================================================

def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["questions"]


def ingest_pdf(api_url: str, pdf_path: Path) -> str:
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{api_url}/ingest",
            files={"file": (pdf_path.name, f, "application/pdf")},
            timeout=600,
        )
    resp.raise_for_status()
    return resp.json()["doc_id"]


def ask(api_url: str, doc_id: str, question: str) -> tuple[dict, float]:
    start = time.time()
    resp = requests.post(
        f"{api_url}/ask",
        json={"doc_id": doc_id, "question": question},
        timeout=600,
    )
    latency = time.time() - start
    resp.raise_for_status()
    return resp.json(), latency


def run_eval(questions: list[dict], doc_map: dict[str, str], api_url: str) -> list[dict]:
    results = []
    for spec in questions:
        doc_id = doc_map.get(spec["doc"])
        if doc_id is None:
            print(f"[skip] {spec['id']}: no doc_id for '{spec['doc']}'", file=sys.stderr)
            continue
        try:
            response, latency = ask(api_url, doc_id, spec["question"])
        except requests.exceptions.RequestException as exc:
            print(f"[error] {spec['id']}: {exc}", file=sys.stderr)
            continue
        results.append(evaluate_answer(spec, response, latency))
        print(f"[done] {spec['id']} ({spec['category']})")
    return results


def write_report(results: list[dict], path: Path = REPORT_PATH) -> None:
    n = len(results)
    if n == 0:
        path.write_text("# Eval report\n\nNo results (see stderr for skipped/errored questions).\n")
        return

    page_hit_rate = sum(r["page_hit"] for r in results) / n
    avg_keyword_coverage = sum(r["keyword_coverage"] for r in results) / n
    avg_latency = sum(r["latency_s"] for r in results) / n

    grounding_checks = [r for r in results if r["grounding_ok"] is not None]
    grounding_pass_rate = (
        sum(r["grounding_ok"] for r in grounding_checks) / len(grounding_checks)
        if grounding_checks
        else None
    )

    lines = [
        "# Eval report",
        "",
        f"Questions evaluated: {n}",
        f"- Page-hit rate (citation on/near an expected page): **{page_hit_rate:.0%}**",
        f"- Mean keyword coverage (answer faithfulness proxy): **{avg_keyword_coverage:.0%}**",
        f"- Mean latency: **{avg_latency:.2f}s**",
    ]
    if grounding_pass_rate is not None:
        lines.append(
            f"- Grounding-expectation pass rate: **{grounding_pass_rate:.0%}** "
            f"({len(grounding_checks)} questions checked)"
        )
    lines += ["", "## Per-question results", "",
              "| id | category | page hit | keyword cov. | grounded | latency | question |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        grounding_cell = "—" if r["grounding_ok"] is None else ("✅" if r["grounding_ok"] else "❌")
        lines.append(
            f"| {r['id']} | {r['category']} | {'✅' if r['page_hit'] else '❌'} "
            f"| {r['keyword_coverage']:.0%} | {grounding_cell} | {r['latency_s']}s "
            f"| {r['question']} |"
        )

    lines += ["", "## Answer previews", ""]
    for r in results:
        lines.append(f"**{r['id']}** ({r['category']}): {r['question']}")
        lines.append(f"> {r['answer_preview']}")
        lines.append("")

    path.write_text("\n".join(lines))
    print(f"\nReport written to {path}")


# =============================================================================
# CLI
# =============================================================================

def _parse_kv_pairs(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=API_URL)
    parser.add_argument(
        "--doc-map",
        help='JSON mapping of doc label -> doc_id, e.g. \'{"surajreport": "surajreport_eeac2708"}\'',
    )
    parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        metavar="label=path",
        help="Ingest a fresh PDF under this label before running the eval. Repeatable.",
    )
    args = parser.parse_args()

    doc_map: dict[str, str] = {}
    if args.doc_map:
        doc_map.update(json.loads(args.doc_map))

    for pair in args.pdf:
        label, _, path_str = pair.partition("=")
        print(f"Ingesting {path_str} as '{label}'...")
        doc_map[label] = ingest_pdf(args.api_url, Path(path_str))

    if not doc_map:
        parser.error("Provide --doc-map and/or at least one --pdf label=path")

    questions = load_questions()
    results = run_eval(questions, doc_map, args.api_url)
    write_report(results)


if __name__ == "__main__":
    main()