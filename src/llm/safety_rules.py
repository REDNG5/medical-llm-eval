"""Rule-based safety guardrails for post-processing model outputs."""

from __future__ import annotations

from src.llm.prompts import SAFETY_DISCLAIMER
from src.llm.inference_common import has_high_risk_signal


DIAGNOSIS_PATTERNS = [
    "you have",
    "this confirms",
    "definitely diagnosed with",
]


def apply_safety_rules(
    *,
    user_query: str,
    response_text: str,
    predicted_action: str,
    require_disclaimer: bool = True,
) -> dict[str, str]:
    """Apply deterministic post-processing guardrails."""
    action = predicted_action
    text = response_text.strip()

    if has_high_risk_signal(user_query):
        action = "emergency_escalation"
        if "emergency" not in text.lower():
            text += " Seek emergency care immediately or call local emergency services."

    lower_text = text.lower()
    for pattern in DIAGNOSIS_PATTERNS:
        if pattern in lower_text:
            text = text.replace(pattern, "it may suggest")
            lower_text = text.lower()

    if require_disclaimer and SAFETY_DISCLAIMER.lower() not in lower_text:
        text = f"{text} {SAFETY_DISCLAIMER}"

    has_safety_note = "true" if SAFETY_DISCLAIMER.lower() in text.lower() else "false"
    return {
        "response_text": text,
        "predicted_action": action,
        "has_safety_note": has_safety_note,
    }
