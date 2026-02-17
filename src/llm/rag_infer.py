"""Enhanced RAG + safety-rule inference runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.llm.inference_common import (
    estimate_confidence,
    infer_action_from_query,
    sanitize_infer_input,
)
from src.llm.prompts import rag_prompt
from src.llm.provider_client import (
    build_structured_output_instructions,
    generate_structured_with_fallback,
)
from src.llm.safety_rules import apply_safety_rules
from src.utils.io import load_yaml, read_csv, write_csv, write_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

PREDICTION_COLUMNS = [
    "sample_id",
    "model_variant",
    "response_text",
    "predicted_action",
    "citations",
    "confidence",
    "has_safety_note",
    "prompt_name",
    "temperature",
]

ALLOWED_ACTIONS = {"inform", "advise_visit", "emergency_escalation", "abstain"}
TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "for",
    "from",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
}


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in text.split():
        token = raw.strip(".,!? ").lower()
        if not token or token in TOKEN_STOPWORDS:
            continue
        if len(token) < 2 and not token.isdigit():
            continue
        tokens.add(token)
    return tokens


def load_knowledge_base(path: str) -> list[dict[str, Any]]:
    """Load and validate external retrieval knowledge base."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("documents", [])
    else:
        entries = []
    if not isinstance(entries, list):
        raise ValueError(f"Knowledge base must be a JSON list or object with 'documents': {path}")

    normalized: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid knowledge entry at index {idx}: must be object.")

        source_id = str(entry.get("id", "")).strip()
        snippet = str(entry.get("snippet", "")).strip()
        raw_keywords = entry.get("keywords", [])
        if isinstance(raw_keywords, str):
            keywords = [raw_keywords.strip()] if raw_keywords.strip() else []
        elif isinstance(raw_keywords, list):
            keywords = [str(k).strip() for k in raw_keywords if str(k).strip()]
        else:
            keywords = []

        if not source_id or not snippet or not keywords:
            raise ValueError(
                f"Invalid knowledge entry at index {idx}: requires non-empty id/snippet/keywords."
            )
        recommended_action = str(entry.get("recommended_action", "inform")).strip().lower()
        if recommended_action not in ALLOWED_ACTIONS:
            raise ValueError(
                f"Invalid knowledge entry at index {idx}: unknown recommended_action '{recommended_action}'."
            )

        normalized.append(
            {
                "id": source_id,
                "snippet": snippet,
                "keywords": keywords,
                "recommended_action": recommended_action,
            }
        )

    if not normalized:
        raise ValueError(f"Knowledge base is empty: {path}")
    return normalized


def retrieve_context(
    query: str, knowledge_base: list[dict[str, Any]], top_k: int = 2
) -> list[dict[str, Any]]:
    """Keyword-overlap retrieval from external guideline snippets."""
    q_tokens = _tokenize(query)
    scored: list[tuple[int, dict[str, Any]]] = []
    for item in knowledge_base:
        score = 0
        for keyword in item["keywords"]:
            parts = _tokenize(keyword)
            if parts and parts.issubset(q_tokens):
                score += 2
            elif parts & q_tokens:
                score += 1
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def _predict_action(row: dict[str, str], retrieved: list[dict[str, Any]]) -> str:
    query = row.get("user_query", "")
    action = infer_action_from_query(query, has_retrieval_support=bool(retrieved))
    retrieved_actions = {str(item.get("recommended_action", "")).strip().lower() for item in retrieved}

    if "emergency_escalation" in retrieved_actions:
        return "emergency_escalation"
    if action == "inform" and "advise_visit" in retrieved_actions:
        return "advise_visit"
    if action == "inform" and "abstain" in retrieved_actions:
        return "abstain"
    return action


def _draft_response(row: dict[str, str], retrieved: list[dict[str, Any]], action: str) -> str:
    query = row.get("user_query", "")
    refs = " ".join(f"[{r['id']}] {r['snippet']}" for r in retrieved)
    if action == "emergency_escalation":
        return (
            f"Your symptoms require urgent in-person assessment. Seek emergency care now. "
            f"Relevant guidance: {refs}"
        )
    if action == "advise_visit":
        return (
            "These symptoms should be reviewed by a clinician soon. "
            f"Interim guidance: monitor closely and avoid risky self-medication. {refs}"
        )
    if action == "abstain":
        return (
            "I cannot safely provide specific triage yet. Please share symptom duration, severity, and red-flag signs. "
            f"General references: {refs}"
        )
    return f"Based on available guidance, supportive care and monitoring are reasonable. {refs} Query: {query}"


def _build_generation_prompt(query: str, retrieved: list[dict[str, Any]]) -> str:
    context = "\n".join(f"[{r['id']}] {r['snippet']}" for r in retrieved)
    return f"{rag_prompt(query, context)}\n\n{build_structured_output_instructions()}"


def run_enhanced(
    input_csv: str,
    output_csv: str,
    metadata_out: str,
    models_config: str,
    seed: int,
) -> None:
    """Run retrieval + guardrail inference and write predictions."""
    set_seed(seed)
    logger = get_logger("rag_infer")
    rows = read_csv(input_csv)
    all_cfg = load_yaml(models_config)
    shared_cfg = all_cfg.get("shared_decoding", {})
    model_cfg = {**shared_cfg, **all_cfg.get("enhanced", {})}
    prompt_name = str(model_cfg.get("prompt_name", "rag_with_safety_guardrails"))
    temperature = float(model_cfg.get("temperature", 0.2))
    top_p = float(model_cfg.get("top_p", 1.0))
    max_tokens = int(model_cfg.get("max_tokens", 300))
    top_k = int(model_cfg.get("retrieval_top_k", 2))
    knowledge_base_path = str(model_cfg.get("knowledge_base_path", "data/raw/knowledge_base.json"))
    knowledge_base = load_knowledge_base(knowledge_base_path)
    logger.info("Loaded knowledge base: %s (%d entries)", knowledge_base_path, len(knowledge_base))
    llm_mode = os.getenv("LLM_MODE", str(model_cfg.get("llm_mode", "mock")))
    provider = "openai" if llm_mode.lower() == "openai" else str(model_cfg.get("provider", "mock"))

    outputs: list[dict[str, str]] = []
    generation_sources: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    api_success_count = 0
    for row in rows:
        infer_row = sanitize_infer_input(row)
        query = infer_row.get("user_query", "")
        retrieved = retrieve_context(query, knowledge_base=knowledge_base, top_k=top_k)
        action = _predict_action(infer_row, retrieved)
        draft = _draft_response(infer_row, retrieved, action)
        prompt = _build_generation_prompt(query, retrieved)
        default_citations = [r["id"] for r in retrieved]
        default_confidence = estimate_confidence(
            query=query,
            action=action,
            response_text=draft,
            citations_count=len(default_citations),
        )
        llm_out = generate_structured_with_fallback(
            prompt=prompt,
            model_cfg=model_cfg,
            default_action=action,
            default_answer=draft,
            default_citations=default_citations,
            default_confidence=default_confidence,
        )
        generation_sources[llm_out.generation_source] += 1
        if llm_out.fallback_reason != "none":
            fallback_reasons[llm_out.fallback_reason] += 1
        if llm_out.api_success:
            api_success_count += 1
        guarded = apply_safety_rules(
            user_query=query,
            response_text=llm_out.response_text,
            predicted_action=llm_out.predicted_action,
            require_disclaimer=True,
        )
        citations = ";".join(llm_out.citations)
        confidence = llm_out.confidence
        outputs.append(
            {
                "sample_id": row.get("sample_id", ""),
                "model_variant": "enhanced",
                "response_text": guarded["response_text"],
                "predicted_action": guarded["predicted_action"],
                "citations": citations,
                "confidence": f"{confidence:.2f}",
                "has_safety_note": guarded["has_safety_note"],
                "prompt_name": prompt_name,
                "temperature": f"{temperature:.2f}",
            }
        )

    write_csv(output_csv, outputs, PREDICTION_COLUMNS)

    sample_context = (
        retrieve_context(rows[0].get("user_query", ""), knowledge_base=knowledge_base, top_k=top_k)
        if rows
        else []
    )
    prompt_preview = _build_generation_prompt(rows[0].get("user_query", "") if rows else "", sample_context)
    prompt_hash = hashlib.sha256(prompt_preview.encode("utf-8")).hexdigest()
    if not generation_sources:
        generation_source = "none"
    elif len(generation_sources) == 1:
        generation_source = next(iter(generation_sources))
    else:
        generation_source = "mixed"
    fallback_reason = (
        "none"
        if not fallback_reasons
        else ";".join(f"{k}:{v}" for k, v in sorted(fallback_reasons.items()))
    )
    write_json(
        metadata_out,
        {
            "variant": "enhanced",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_csv": input_csv,
            "output_csv": output_csv,
            "prompt_name": prompt_name,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "provider": provider,
            "llm_mode": llm_mode,
            "model_name": str(model_cfg.get("model_name", "")),
            "retrieval_top_k": top_k,
            "knowledge_base_path": knowledge_base_path,
            "knowledge_base_size": len(knowledge_base),
            "prompt_hash": prompt_hash,
            "generation_source": generation_source,
            "fallback_reason": fallback_reason,
            "api_success_count": api_success_count,
            "seed": seed,
            "prompt_preview": prompt_preview,
        },
    )
    logger.info("Enhanced predictions written: %s (%d rows)", output_csv, len(outputs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run enhanced RAG+guardrail inference.")
    parser.add_argument("--input_csv", default="data/processed/eval_samples.csv")
    parser.add_argument("--output_csv", default="reports/tables/predictions_enhanced.csv")
    parser.add_argument("--metadata_out", default="reports/tables/run_metadata_enhanced.json")
    parser.add_argument("--models_config", default="configs/models.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_enhanced(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        metadata_out=args.metadata_out,
        models_config=args.models_config,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
