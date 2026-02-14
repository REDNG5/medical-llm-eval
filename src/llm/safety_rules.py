"""Rule-based safety guardrails for post-processing model outputs."""

from __future__ import annotations

from src.llm.prompts import SAFETY_DISCLAIMER
from src.utils.io import parse_bool

HIGH_RISK_KEYWORDS = {
    "chest pain",
    "shortness of breath",
    "difficult to wake",
    "bleeding",
    "unconscious",
    "severe abdominal pain",
    "confused",
}

DIAGNOSIS_PATTERNS = [
    "you have",
    "this confirms",
    "definitely diagnosed with",
]


def _contains_high_risk_signal(query: str) -> bool:
    q = query.lower()
    return any(keyword in q for keyword in HIGH_RISK_KEYWORDS)


def apply_safety_rules(
    *,
    user_query: str,
    risk_level: str,
    required_safety_note: str | bool,
    response_text: str,
    predicted_action: str,
) -> dict[str, str]:
    """Apply deterministic post-processing guardrails."""
    action = predicted_action
    text = response_text.strip()

    if risk_level == "high" or _contains_high_risk_signal(user_query):
        action = "emergency_escalation"
        if "emergency" not in text.lower():
            text += " Seek emergency care immediately or call local emergency services."

    lower_text = text.lower()
    for pattern in DIAGNOSIS_PATTERNS:
        if pattern in lower_text:
            text = text.replace(pattern, "it may suggest")
            lower_text = text.lower()

    if parse_bool(required_safety_note) and SAFETY_DISCLAIMER.lower() not in lower_text:
        text = f"{text} {SAFETY_DISCLAIMER}"

    has_safety_note = "true" if SAFETY_DISCLAIMER.lower() in text.lower() else "false"
    return {
        "response_text": text,
        "predicted_action": action,
        "has_safety_note": has_safety_note,
    }

