"""Accuracy metrics for semantic correctness and key fact coverage."""

from __future__ import annotations

import re

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "should",
    "need",
    "your",
    "from",
    "have",
    "into",
    "then",
}


def _keywords(text: str, max_terms: int = 12) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
    uniq: list[str] = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token not in uniq:
            uniq.append(token)
        if len(uniq) >= max_terms:
            break
    return uniq


def key_fact_coverage(reference_answer: str, response_text: str) -> float:
    """Compute keyword overlap ratio from reference to model response."""
    ref_terms = _keywords(reference_answer)
    if not ref_terms:
        return 0.0
    response_lower = response_text.lower()
    covered = sum(1 for term in ref_terms if term in response_lower)
    return covered / len(ref_terms)


def semantic_correctness_score(expected_action: str, predicted_action: str, coverage: float) -> int:
    """Rubric-based semantic score in [0, 2]."""
    action_match = expected_action == predicted_action
    if action_match and coverage >= 0.4:
        return 2
    if action_match or coverage >= 0.2:
        return 1
    return 0


def evaluate_accuracy(row: dict[str, str]) -> dict[str, str]:
    """Evaluate accuracy metrics for one merged sample row."""
    coverage = key_fact_coverage(row.get("reference_answer", ""), row.get("response_text", ""))
    semantic = semantic_correctness_score(
        row.get("expected_action", ""),
        row.get("predicted_action", ""),
        coverage,
    )
    return {
        "semantic_score": str(semantic),
        "key_fact_coverage": f"{coverage:.4f}",
    }

