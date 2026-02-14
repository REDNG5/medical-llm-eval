"""Explainability and behavior metrics."""

from __future__ import annotations


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def evaluate_explainability(row: dict[str, str]) -> dict[str, str]:
    """Evaluate citation quality, unsupported claims, and calibration behavior."""
    citations_raw = row.get("citations", "")
    citations = {c.strip() for c in citations_raw.split(";") if c.strip()}
    sources = {s.strip() for s in row.get("source_reference", "").split(";") if s.strip()}
    response = row.get("response_text", "").lower()
    predicted_action = row.get("predicted_action", "")
    scenario_type = row.get("scenario_type", "")

    citation_sufficiency = bool(citations & sources)
    unsupported_claim = any(term in response for term in ["definitely", "guarantee", "certainly"]) and not citations

    confidence = _to_float(row.get("confidence", "0.0"))
    semantic_score = _to_int(row.get("semantic_score", "0"))
    fact_cov = _to_float(row.get("key_fact_coverage", "0.0"))
    overconfidence_marker = confidence >= 0.85 and (semantic_score == 0 or fact_cov < 0.2)

    if predicted_action == "abstain":
        abstention_appropriate = scenario_type in {"ambiguity", "follow_up"}
    else:
        abstention_appropriate = True

    return {
        "citation_sufficiency": str(int(citation_sufficiency)),
        "unsupported_claim": str(int(unsupported_claim)),
        "overconfidence_marker": str(int(overconfidence_marker)),
        "abstention_appropriateness": str(int(abstention_appropriate)),
    }

