"""Shared inference helpers that avoid label leakage."""

from __future__ import annotations

from src.llm.prompts import SAFETY_DISCLAIMER

ALLOWED_INFER_KEYS = {"sample_id", "user_query"}

HIGH_RISK_SIGNALS = {
    "chest pain",
    "shortness of breath",
    "difficulty breathing",
    "difficult to wake",
    "unconscious",
    "confused and difficult to wake",
    "severe abdominal pain",
    "bleeding won't stop",
    "bleeding wont stop",
}

AMBIGUITY_SIGNALS = {
    "not sure",
    "unclear",
    "maybe",
    "dont know",
    "don't know",
    "unsure",
}

ADVISE_VISIT_SIGNALS = {
    "fever",
    "vomiting",
    "medication",
    "ibuprofen",
    "antibiotic",
    "rash",
    "dizzy",
    "stomach pain",
}


def sanitize_infer_input(row: dict[str, str]) -> dict[str, str]:
    """Keep only fields that inference is allowed to access."""
    return {k: row.get(k, "") for k in ALLOWED_INFER_KEYS}


def has_high_risk_signal(query: str) -> bool:
    """Detect clear high-risk red flags from query text."""
    q = query.lower()
    return any(signal in q for signal in HIGH_RISK_SIGNALS)


def infer_action_from_query(query: str, has_retrieval_support: bool) -> str:
    """Infer triage action from query text plus optional retrieval support."""
    q = query.lower()
    if has_high_risk_signal(q):
        return "emergency_escalation"
    if any(signal in q for signal in AMBIGUITY_SIGNALS):
        return "abstain"
    if any(signal in q for signal in ADVISE_VISIT_SIGNALS):
        return "advise_visit"
    if not has_retrieval_support and ("what should i do" in q or "what can i do" in q):
        return "inform"
    return "inform"


def has_safety_note(text: str) -> bool:
    """Check whether the standard disclaimer is present."""
    return SAFETY_DISCLAIMER.lower() in text.lower()


def estimate_confidence(
    *,
    query: str,
    action: str,
    response_text: str,
    citations_count: int,
) -> float:
    """Compute deterministic confidence proxy from observable signals."""
    q = query.lower()
    response = response_text.lower()
    score = 0.55

    if citations_count > 0:
        score += 0.18
    else:
        score -= 0.08

    if action in {"abstain", "emergency_escalation"}:
        score += 0.07
    if any(token in q for token in AMBIGUITY_SIGNALS):
        score -= 0.13
    if any(token in response for token in {"may", "might", "could"}):
        score -= 0.06
    if action == "inform" and citations_count == 0:
        score -= 0.06

    if has_high_risk_signal(q) and action != "emergency_escalation":
        score -= 0.25
    if has_safety_note(response_text):
        score += 0.03

    return round(max(0.05, min(0.95, score)), 2)

