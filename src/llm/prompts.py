"""Prompt templates for baseline and enhanced inference."""

from __future__ import annotations

SAFETY_DISCLAIMER = (
    "This is general educational information, not a diagnosis. "
    "If symptoms are severe or worsening, seek in-person medical care."
)


def baseline_prompt(user_query: str) -> str:
    """Prompt used by baseline direct-answer model."""
    return (
        "You are a medical information assistant. "
        "Answer briefly for symptom triage support without diagnosis.\n"
        f"User question: {user_query}\n"
        "Output: guidance and next-step recommendation."
    )


def rag_prompt(user_query: str, retrieved_context: str) -> str:
    """Prompt used by enhanced RAG+guardrail model."""
    return (
        "You are a cautious medical information assistant.\n"
        "Use retrieved references, avoid diagnosis claims, and escalate red flags.\n"
        f"Retrieved context:\n{retrieved_context}\n\n"
        f"User question: {user_query}\n"
        "Output: concise guidance, action label, and citations."
    )

