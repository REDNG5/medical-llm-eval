"""Baseline (no retrieval) inference runner."""

from __future__ import annotations

import argparse
import hashlib
import os
from collections import Counter
from datetime import datetime, timezone

from src.llm.inference_common import (
    estimate_confidence,
    infer_action_from_query,
    sanitize_infer_input,
)
from src.llm.prompts import baseline_prompt
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


def _predict_action_baseline(infer_row: dict[str, str]) -> str:
    """Predict action using query-only signals."""
    query = infer_row.get("user_query", "")
    return infer_action_from_query(query, has_retrieval_support=False)


def _generate_response(infer_row: dict[str, str], action: str) -> str:
    query = infer_row.get("user_query", "")

    if action == "emergency_escalation":
        text = (
            f"Based on your symptoms ({query}), this could be urgent. "
            "Please seek emergency care now."
        )
    elif action == "advise_visit":
        text = (
            f"This may need a clinician review soon. For now, monitor symptoms and arrange a visit."
        )
    elif action == "abstain":
        text = "I do not have enough details to give safe guidance. Please share more specifics."
    else:
        text = "This appears mild. Rest, hydration, and symptom monitoring are reasonable first steps."

    return text


def _build_generation_prompt(query: str) -> str:
    return (
        f"{baseline_prompt(query)}\n\n"
        f"{build_structured_output_instructions()}"
    )


def run_baseline(
    input_csv: str,
    output_csv: str,
    metadata_out: str,
    models_config: str,
    seed: int,
) -> None:
    """Run deterministic baseline inference and write predictions."""
    set_seed(seed)
    logger = get_logger("baseline_infer")
    rows = read_csv(input_csv)
    all_cfg = load_yaml(models_config)
    shared_cfg = all_cfg.get("shared_decoding", {})
    model_cfg = {**shared_cfg, **all_cfg.get("baseline", {})}
    prompt_name = str(model_cfg.get("prompt_name", "baseline_direct_answer"))
    temperature = float(model_cfg.get("temperature", 0.2))
    top_p = float(model_cfg.get("top_p", 1.0))
    max_tokens = int(model_cfg.get("max_tokens", 300))
    llm_mode = os.getenv("LLM_MODE", str(model_cfg.get("llm_mode", "mock")))
    provider = "openai" if llm_mode.lower() == "openai" else str(model_cfg.get("provider", "mock"))

    outputs: list[dict[str, str]] = []
    generation_sources: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    api_success_count = 0
    for row in rows:
        infer_row = sanitize_infer_input(row)
        action = _predict_action_baseline(infer_row)
        response_text = _generate_response(infer_row, action)
        default_confidence = estimate_confidence(
            query=infer_row.get("user_query", ""),
            action=action,
            response_text=response_text,
            citations_count=0,
        )
        prompt = _build_generation_prompt(infer_row.get("user_query", ""))
        llm_out = generate_structured_with_fallback(
            prompt=prompt,
            model_cfg=model_cfg,
            default_action=action,
            default_answer=response_text,
            default_citations=[],
            default_confidence=default_confidence,
        )
        generation_sources[llm_out.generation_source] += 1
        if llm_out.fallback_reason != "none":
            fallback_reasons[llm_out.fallback_reason] += 1
        if llm_out.api_success:
            api_success_count += 1
        guarded = apply_safety_rules(
            user_query=infer_row.get("user_query", ""),
            response_text=llm_out.response_text,
            predicted_action=llm_out.predicted_action,
            require_disclaimer=True,
        )
        citations = ";".join(llm_out.citations)
        confidence = llm_out.confidence
        outputs.append(
            {
                "sample_id": row.get("sample_id", ""),
                "model_variant": "baseline",
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
    if rows:
        prompt_preview = _build_generation_prompt(rows[0].get("user_query", ""))
    else:
        prompt_preview = _build_generation_prompt("")
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
            "variant": "baseline",
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
            "prompt_hash": prompt_hash,
            "generation_source": generation_source,
            "fallback_reason": fallback_reason,
            "api_success_count": api_success_count,
            "seed": seed,
            "prompt_preview": prompt_preview,
        },
    )
    logger.info("Baseline predictions written: %s (%d rows)", output_csv, len(outputs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline no-retrieval inference.")
    parser.add_argument("--input_csv", default="data/processed/eval_samples.csv")
    parser.add_argument("--output_csv", default="reports/tables/predictions_baseline.csv")
    parser.add_argument("--metadata_out", default="reports/tables/run_metadata_baseline.json")
    parser.add_argument("--models_config", default="configs/models.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_baseline(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        metadata_out=args.metadata_out,
        models_config=args.models_config,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
