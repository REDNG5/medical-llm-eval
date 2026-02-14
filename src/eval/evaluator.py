"""End-to-end evaluator: merge predictions with labels and compute metrics."""

from __future__ import annotations

import argparse
from collections import Counter

from src.eval.error_taxonomy import detect_error_tags, taxonomy_rows
from src.eval.metrics_accuracy import evaluate_accuracy
from src.eval.metrics_explainability import evaluate_explainability
from src.eval.metrics_safety import evaluate_safety
from src.utils.io import ensure_dir, read_csv, write_csv
from src.utils.logging import get_logger

PER_SAMPLE_FIELDS = [
    "sample_id",
    "split",
    "scenario_type",
    "risk_level",
    "expected_action",
    "predicted_action",
    "response_text",
    "source_reference",
    "citations",
    "confidence",
    "semantic_score",
    "key_fact_coverage",
    "unsafe_advice",
    "high_risk_miss",
    "escalation_compliance",
    "safety_note_compliance",
    "citation_sufficiency",
    "unsupported_claim",
    "overconfidence_marker",
    "abstention_appropriateness",
    "error_tags",
    "model_variant",
]

METRIC_FIELDS = [
    "model_variant",
    "num_samples",
    "semantic_score_mean",
    "semantic_pass_rate",
    "key_fact_coverage_mean",
    "unsafe_advice_rate",
    "high_risk_miss_rate",
    "escalation_compliance_rate",
    "safety_note_compliance_rate",
    "citation_sufficiency_rate",
    "unsupported_claim_rate",
    "overconfidence_rate",
    "abstention_appropriateness_rate",
    "top_error_type",
]


def _to_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate_from_flag(rows: list[dict[str, str]], flag: str) -> float:
    if not rows:
        return 0.0
    positives = sum(_to_int(row.get(flag, "0")) for row in rows)
    return positives / len(rows)


def evaluate_predictions(eval_csv: str, pred_csv: str, output_dir: str) -> tuple[str, str]:
    """Evaluate one prediction file and return output metric/per-sample paths."""
    logger = get_logger("evaluator")
    labels = {row["sample_id"]: row for row in read_csv(eval_csv)}
    preds = read_csv(pred_csv)
    merged_rows: list[dict[str, str]] = []

    for pred in preds:
        sample_id = pred.get("sample_id", "")
        label = labels.get(sample_id)
        if not label:
            continue
        row = {**label, **pred}
        row.update(evaluate_accuracy(row))
        row.update(evaluate_safety(row))
        row.update(evaluate_explainability(row))
        row["error_tags"] = ";".join(detect_error_tags(row))
        merged_rows.append(row)

    if not merged_rows:
        raise ValueError("No merged rows found. Check sample_id consistency.")

    variant = merged_rows[0].get("model_variant", "unknown")
    out_dir = ensure_dir(output_dir)
    per_sample_path = str(out_dir / f"per_sample_eval_{variant}.csv")
    metric_path = str(out_dir / f"metrics_{variant}.csv")
    taxonomy_path = str(out_dir / "error_taxonomy_reference.csv")

    write_csv(
        per_sample_path,
        [{k: row.get(k, "") for k in PER_SAMPLE_FIELDS} for row in merged_rows],
        PER_SAMPLE_FIELDS,
    )

    semantic_scores = [_to_int(r.get("semantic_score", "0")) for r in merged_rows]
    semantic_pass_rate = sum(1 for s in semantic_scores if s >= 1) / len(semantic_scores)
    key_cov = [_to_float(r.get("key_fact_coverage", "0")) for r in merged_rows]

    error_counter: Counter[str] = Counter()
    for row in merged_rows:
        for tag in [t for t in row.get("error_tags", "").split(";") if t]:
            error_counter[tag] += 1
    top_error = error_counter.most_common(1)[0][0] if error_counter else "none"

    metrics_row = {
        "model_variant": variant,
        "num_samples": str(len(merged_rows)),
        "semantic_score_mean": f"{_mean([float(s) for s in semantic_scores]):.4f}",
        "semantic_pass_rate": f"{semantic_pass_rate:.4f}",
        "key_fact_coverage_mean": f"{_mean(key_cov):.4f}",
        "unsafe_advice_rate": f"{_rate_from_flag(merged_rows, 'unsafe_advice'):.4f}",
        "high_risk_miss_rate": f"{_rate_from_flag(merged_rows, 'high_risk_miss'):.4f}",
        "escalation_compliance_rate": f"{_rate_from_flag(merged_rows, 'escalation_compliance'):.4f}",
        "safety_note_compliance_rate": f"{_rate_from_flag(merged_rows, 'safety_note_compliance'):.4f}",
        "citation_sufficiency_rate": f"{_rate_from_flag(merged_rows, 'citation_sufficiency'):.4f}",
        "unsupported_claim_rate": f"{_rate_from_flag(merged_rows, 'unsupported_claim'):.4f}",
        "overconfidence_rate": f"{_rate_from_flag(merged_rows, 'overconfidence_marker'):.4f}",
        "abstention_appropriateness_rate": f"{_rate_from_flag(merged_rows, 'abstention_appropriateness'):.4f}",
        "top_error_type": top_error,
    }

    write_csv(metric_path, [metrics_row], METRIC_FIELDS)
    write_csv(
        taxonomy_path,
        taxonomy_rows(),
        ["error_type", "detection_rule", "example", "likely_cause", "remediation_action"],
    )

    logger.info("Evaluation complete for %s", variant)
    logger.info("Per-sample: %s", per_sample_path)
    logger.info("Summary metrics: %s", metric_path)
    return per_sample_path, metric_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument("--eval_csv", default="data/processed/eval_samples.csv")
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--output_dir", default="reports/tables")
    args = parser.parse_args()
    evaluate_predictions(args.eval_csv, args.pred_csv, args.output_dir)


if __name__ == "__main__":
    main()

