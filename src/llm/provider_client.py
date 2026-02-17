"""LLM provider client with mock fallback support."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any


ALLOWED_ACTIONS = {"inform", "advise_visit", "emergency_escalation", "abstain"}


@dataclass
class LLMStructuredOutput:
    """Structured LLM output used by inference pipelines."""

    response_text: str
    predicted_action: str
    confidence: float
    citations: list[str]
    raw_text: str
    generation_source: str
    fallback_reason: str
    api_success: bool


def build_structured_output_instructions() -> str:
    """Instruction block for predictable parseable output."""
    return (
        "Return output using this exact format:\n"
        "ACTION: <inform|advise_visit|emergency_escalation|abstain>\n"
        "CONFIDENCE: <0.00-1.00>\n"
        "CITATIONS: <semicolon-separated source ids or NONE>\n"
        "ANSWER: <concise clinical-safety-oriented response>\n"
        "Do not include diagnosis claims."
    )


def generate_structured_with_fallback(
    *,
    prompt: str,
    model_cfg: dict[str, Any],
    default_action: str,
    default_answer: str,
    default_citations: list[str] | None = None,
    default_confidence: float = 0.6,
) -> LLMStructuredOutput:
    """Generate model output in real or mock mode with robust fallback."""
    mode = os.getenv("LLM_MODE", str(model_cfg.get("llm_mode", "mock"))).lower()
    if mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return _fallback_output(
                default_action=default_action,
                default_answer=default_answer,
                default_citations=default_citations,
                default_confidence=default_confidence,
                raw_text="FALLBACK: OPENAI_API_KEY not set.",
                generation_source="fallback",
                fallback_reason="missing_openai_api_key",
                api_success=False,
            )
        try:
            raw = _generate_openai(prompt, model_cfg, api_key)
            parsed = _parse_structured_output(
                raw,
                default_action=default_action,
                default_answer=default_answer,
                default_citations=default_citations or [],
                default_confidence=default_confidence,
            )
            return parsed
        except Exception as exc:  # noqa: BLE001
            return _fallback_output(
                default_action=default_action,
                default_answer=default_answer,
                default_citations=default_citations,
                default_confidence=default_confidence,
                raw_text=f"FALLBACK: OpenAI call failed: {exc}",
                generation_source="fallback",
                fallback_reason=f"openai_error:{exc.__class__.__name__}",
                api_success=False,
            )

    return _fallback_output(
        default_action=default_action,
        default_answer=default_answer,
        default_citations=default_citations,
        default_confidence=default_confidence,
        raw_text="MOCK_MODE",
        generation_source="mock",
        fallback_reason="none",
        api_success=False,
    )


def _generate_openai(prompt: str, model_cfg: dict[str, Any], api_key: str) -> str:
    """Call OpenAI Responses API via official client."""
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("openai package not installed. Install with `pip install openai`.") from exc

    model_name = str(model_cfg.get("model_name", "gpt-4o-mini"))
    temperature = float(model_cfg.get("temperature", 0.2))
    top_p = float(model_cfg.get("top_p", 1.0))
    max_tokens = int(model_cfg.get("max_tokens", 300))

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    text = getattr(response, "output_text", "") or ""
    if text.strip():
        return text

    # Defensive fallback if SDK shape differs.
    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for c in content:
                c_text = getattr(c, "text", None)
                if isinstance(c_text, str) and c_text.strip():
                    chunks.append(c_text)
        if chunks:
            return "\n".join(chunks)
    raise RuntimeError("OpenAI response contained no text output.")


def _parse_structured_output(
    raw_text: str,
    *,
    default_action: str,
    default_answer: str,
    default_citations: list[str],
    default_confidence: float,
) -> LLMStructuredOutput:
    """Parse ACTION/CONFIDENCE/CITATIONS/ANSWER text protocol."""
    action_match = re.search(r"(?im)^\s*ACTION:\s*(.+)\s*$", raw_text)
    conf_match = re.search(r"(?im)^\s*CONFIDENCE:\s*([0-9]*\.?[0-9]+)\s*$", raw_text)
    cite_match = re.search(r"(?im)^\s*CITATIONS:\s*(.+)\s*$", raw_text)
    answer_match = re.search(r"(?is)^\s*ANSWER:\s*(.+)$", raw_text, flags=re.MULTILINE)

    action = (action_match.group(1).strip().lower() if action_match else default_action).strip()
    if action not in ALLOWED_ACTIONS:
        action = default_action

    confidence = default_confidence
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except ValueError:
            confidence = default_confidence
    confidence = max(0.0, min(1.0, confidence))

    citations: list[str] = list(default_citations)
    if cite_match:
        c_raw = cite_match.group(1).strip()
        if c_raw.lower() == "none":
            citations = []
        else:
            citations = [c.strip() for c in c_raw.replace(",", ";").split(";") if c.strip()]

    answer = answer_match.group(1).strip() if answer_match else default_answer
    if not answer:
        answer = default_answer

    return LLMStructuredOutput(
        response_text=answer,
        predicted_action=action,
        confidence=confidence,
        citations=citations,
        raw_text=raw_text,
        generation_source="openai",
        fallback_reason="none",
        api_success=True,
    )


def _fallback_output(
    *,
    default_action: str,
    default_answer: str,
    default_citations: list[str] | None,
    default_confidence: float,
    raw_text: str,
    generation_source: str,
    fallback_reason: str,
    api_success: bool,
) -> LLMStructuredOutput:
    """Return deterministic fallback output in mock mode or failure paths."""
    return LLMStructuredOutput(
        response_text=default_answer,
        predicted_action=default_action,
        confidence=max(0.0, min(1.0, float(default_confidence))),
        citations=list(default_citations or []),
        raw_text=raw_text,
        generation_source=generation_source,
        fallback_reason=fallback_reason,
        api_success=api_success,
    )
