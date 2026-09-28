"""
Unit tests for the pure scoring functions in notebooks/eval_retrieval.py.

No network calls here - these test the scoring logic in isolation using
synthetic citations/answers, so they run instantly and don't need a live
backend or Ollama.
"""

from eval_retrieval import check_grounding, evaluate_answer, keyword_coverage, page_hit


# ---------------------------------------------------------------------------
# page_hit
# ---------------------------------------------------------------------------

def test_page_hit_true_on_exact_match():
    citations = [{"page": 25, "type": "text"}, {"page": 3, "type": "image"}]
    assert page_hit(citations, expected_pages=[25])


def test_page_hit_false_when_no_page_matches():
    citations = [{"page": 3}, {"page": 9}]
    assert not page_hit(citations, expected_pages=[25])


def test_page_hit_respects_tolerance():
    citations = [{"page": 26}]
    assert page_hit(citations, expected_pages=[25], tolerance=1)
    assert not page_hit(citations, expected_pages=[25], tolerance=0)


def test_page_hit_true_when_no_expected_pages_given():
    # An empty expected_pages list means "nothing specific to check" -
    # used for grounding-only questions - and should not count as a miss.
    assert page_hit([{"page": 1}], expected_pages=[])
    assert page_hit([], expected_pages=[])


def test_page_hit_ignores_citations_without_a_page():
    citations = [{"type": "text"}]  # malformed/missing page
    assert not page_hit(citations, expected_pages=[25])


# ---------------------------------------------------------------------------
# keyword_coverage
# ---------------------------------------------------------------------------

def test_keyword_coverage_full_match():
    answer = "The encoder uses residual blocks with attention and skip connections."
    assert keyword_coverage(answer, ["encoder", "attention", "residual"]) == 1.0


def test_keyword_coverage_partial_match():
    answer = "The encoder uses residual blocks."
    coverage = keyword_coverage(answer, ["encoder", "decoder"])
    assert coverage == 0.5


def test_keyword_coverage_is_case_insensitive():
    answer = "DFPIR++ IS AN ENCODER-DECODER MODEL."
    assert keyword_coverage(answer, ["dfpir++", "encoder-decoder"]) == 1.0


def test_keyword_coverage_empty_keyword_list_is_trivially_satisfied():
    assert keyword_coverage("anything at all", []) == 1.0


def test_keyword_coverage_zero_when_nothing_matches():
    assert keyword_coverage("completely unrelated text", ["dfpir++", "encoder"]) == 0.0


# ---------------------------------------------------------------------------
# check_grounding
# ---------------------------------------------------------------------------

def test_check_grounding_none_when_no_expectation_set():
    assert check_grounding(actual_grounded=True, expect_grounded=None) is None


def test_check_grounding_true_when_matches_expectation():
    assert check_grounding(actual_grounded=True, expect_grounded=True) is True
    assert check_grounding(actual_grounded=False, expect_grounded=False) is True


def test_check_grounding_false_when_mismatched():
    assert check_grounding(actual_grounded=True, expect_grounded=False) is False


# ---------------------------------------------------------------------------
# evaluate_answer (integration of the pure functions above)
# ---------------------------------------------------------------------------

def test_evaluate_answer_combines_all_checks():
    question_spec = {
        "id": "q7",
        "doc": "surajreport",
        "category": "architecture",
        "question": "Explain the DFPIR++ architecture.",
        "expected_pages": [25],
        "expected_keywords": ["encoder-decoder", "degradation"],
    }
    response = {
        "answer": "DFPIR++ is a lightweight encoder-decoder model with degradation "
        "conditioning.",
        "grounded": True,
        "citations": [{"page": 25, "type": "text"}],
    }

    result = evaluate_answer(question_spec, response, latency_s=1.234)

    assert result["id"] == "q7"
    assert result["page_hit"] is True
    assert result["keyword_coverage"] == 1.0
    assert result["grounding_ok"] is None  # no expect_grounded set for this question
    assert result["latency_s"] == 1.23
    assert result["answer_preview"] == response["answer"]


def test_evaluate_answer_truncates_long_previews():
    question_spec = {
        "id": "q1",
        "doc": "surajreport",
        "category": "factual",
        "question": "x",
        "expected_pages": [],
        "expected_keywords": [],
    }
    long_answer = "a" * 300
    response = {"answer": long_answer, "grounded": True, "citations": []}

    result = evaluate_answer(question_spec, response, latency_s=0.5)

    assert len(result["answer_preview"]) == 181  # 180 chars + ellipsis
    assert result["answer_preview"].endswith("…")


def test_evaluate_answer_checks_grounding_expectation_when_present():
    question_spec = {
        "id": "q12",
        "doc": "surajreport",
        "category": "grounding",
        "question": "What is the capital of France?",
        "expected_pages": [],
        "expected_keywords": [],
        "expect_grounded": False,
    }
    response = {"answer": "The document does not mention this.", "grounded": False, "citations": []}

    result = evaluate_answer(question_spec, response, latency_s=0.8)

    assert result["grounding_ok"] is True