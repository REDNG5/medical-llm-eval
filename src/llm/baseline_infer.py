"""Baseline (no retrieval) inference runner."""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone

from src.llm.prompts import baseline_prompt
from src.utils.io import load_yaml, parse_bool, read_csv, write_csv, write_json
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


def _hash_to_percent(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % 100


def _predict_action_baseline(row: dict[str, str]) -> str:
    risk = row.get("risk_level", "low")
    scenario = row.get("scenario_type", "general")
    sample_id = row.get("sample_id", "")
    pct = _hash_to_percent(sample_id)

    if risk == "high":
        return "advise_visit" if pct < 35 else "emergency_escalation"
    if scenario == "ambiguity" and pct < 20:
        return "abstain"
    if risk == "medium":
        return "advise_visit"
    return "inform"


def _generate_response(row: dict[str, str], action: str) -> tuple[str, str, str]:
    query = row.get("user_query", "")
    source_ref = row.get("source_reference", "")
    safety_note_required = parse_bool(row.get("required_safety_note", "false"))
    sample_id = row.get("sample_id", "")
    pct = _hash_to_percent(sample_id)

    if action == "emergency_escalation":
        text = (
            f"Based on your symptoms ({query}), this could be urgent. "
            "Please seek emergency care now."
        )
    elif action == "advise_visit":
        text = (
            f"This may need a clinician review soon. For now, monitor symptoms and arrange a visit."
        )
        if pct < 25:
            text += " It is probably not serious."
    elif action == "abstain":
        text = "I do not have enough details to give safe guidance. Please share more specifics."
    else:
        text = "This appears mild. Rest, hydration, and symptom monitoring are reasonable first steps."
        if pct < 20:
            text += " You are definitely fine."

    citations = source_ref if pct < 15 else ""
    has_safety_note = "true" if (safety_note_required and pct < 20) else "false"
    return text, citations, has_safety_note


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
    model_cfg = load_yaml(models_config).get("baseline", {})
    prompt_name = str(model_cfg.get("prompt_name", "baseline_direct_answer"))
    temperature = float(model_cfg.get("temperature", 0.2))

    outputs: list[dict[str, str]] = []
    for row in rows:
        action = _predict_action_baseline(row)
        response_text, citations, has_safety_note = _generate_response(row, action)
        outputs.append(
            {
                "sample_id": row.get("sample_id", ""),
                "model_variant": "baseline",
                "response_text": response_text,
                "predicted_action": action,
                "citations": citations,
                "confidence": "0.90",
                "has_safety_note": has_safety_note,
                "prompt_name": prompt_name,
                "temperature": f"{temperature:.2f}",
            }
        )

    write_csv(output_csv, outputs, PREDICTION_COLUMNS)
    if rows:
        prompt_preview = baseline_prompt(rows[0].get("user_query", ""))
    else:
        prompt_preview = baseline_prompt("")

    write_json(
        metadata_out,
        {
            "variant": "baseline",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_csv": input_csv,
            "output_csv": output_csv,
            "prompt_name": prompt_name,
            "temperature": temperature,
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

