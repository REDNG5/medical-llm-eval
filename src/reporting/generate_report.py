"""Generate final markdown reports from computed metrics."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.utils.io import ensure_dir, load_yaml, read_csv


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _compare_direction(base_value: float, enhanced_value: float, eps: float = 1e-9) -> str:
    if enhanced_value < base_value - eps:
        return "decrease"
    if enhanced_value > base_value + eps:
        return "increase"
    return "flat"


def _improvement_sentence(metric_name: str, base_value: float, enhanced_value: float, lower_is_better: bool) -> str:
    direction = _compare_direction(base_value, enhanced_value)
    if direction == "flat":
        return f"{metric_name} was unchanged ({base_value:.4f} -> {enhanced_value:.4f})."

    improved = (direction == "decrease" and lower_is_better) or (direction == "increase" and not lower_is_better)
    status = "improved" if improved else "degraded"
    return f"{metric_name} {status} ({base_value:.4f} -> {enhanced_value:.4f})."


def _safe_int(value: str | int | None, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def generate_reports(
    tables_dir: str,
    reports_dir: str,
    eval_csv: str,
    eval_config: str,
) -> tuple[str, str]:
    """Generate final report and deployment one-pager."""
    metrics_path = Path(tables_dir) / "metrics_summary.csv"
    slice_path = Path(tables_dir) / "slice_metrics.csv"
    metrics_rows = read_csv(metrics_path) if metrics_path.exists() else []
    slice_rows = read_csv(slice_path) if slice_path.exists() else []
    eval_rows = read_csv(eval_csv) if Path(eval_csv).exists() else []
    eval_cfg = load_yaml(eval_config) if Path(eval_config).exists() else {}

    by_variant = {row["model_variant"]: row for row in metrics_rows}
    base = by_variant.get("baseline", {})
    enh = by_variant.get("enhanced", {})
    eval_split = str(enh.get("eval_split", base.get("eval_split", "test")))

    total_eval_samples = len(eval_rows)
    split_eval_samples = _safe_int(enh.get("num_samples", base.get("num_samples")), default=0)

    semantic_base = _to_float(base.get("semantic_score_mean", "0"))
    semantic_enh = _to_float(enh.get("semantic_score_mean", "0"))
    high_risk_miss_base = _to_float(base.get("high_risk_miss_rate", "0"))
    high_risk_miss_enh = _to_float(enh.get("high_risk_miss_rate", "0"))
    citation_base = _to_float(base.get("citation_sufficiency_rate", "0"))
    citation_enh = _to_float(enh.get("citation_sufficiency_rate", "0"))

    out_dir = ensure_dir(reports_dir)
    final_path = out_dir / "final_report.md"
    onepager_path = out_dir / "deployment_risk_onepager.md"

    final_report = f"""# Final Report: Medical LLM Evaluation MVP

Date: {date.today().isoformat()}

## Setup
- Pipeline type: Baseline vs Enhanced (RAG + safety rules)
- Eval set size: total={total_eval_samples if total_eval_samples else "n/a"}, {eval_split}={split_eval_samples if split_eval_samples else "n/a"}
- Task scope: medical information and symptom triage support (non-diagnostic)

## Methodology
- Data: semi-automatic evaluation set with manual-review-ready template.
- Baseline: direct answer without retrieval.
- Enhanced: retrieval-augmented drafting plus deterministic safety guardrails.
- Metrics: accuracy, safety, explainability, calibration behavior.

## Metrics
- Semantic score mean (baseline -> enhanced): {base.get("semantic_score_mean", "n/a")} -> {enh.get("semantic_score_mean", "n/a")}
- Unsafe advice rate (baseline -> enhanced): {base.get("unsafe_advice_rate", "n/a")} -> {enh.get("unsafe_advice_rate", "n/a")}
- Citation sufficiency rate (baseline -> enhanced): {base.get("citation_sufficiency_rate", "n/a")} -> {enh.get("citation_sufficiency_rate", "n/a")}
- Citation requirement compliance (baseline -> enhanced): {base.get("citation_requirement_compliance_rate", "n/a")} -> {enh.get("citation_requirement_compliance_rate", "n/a")}
- Forbidden-claim violation rate (baseline -> enhanced): {base.get("forbidden_claim_violation_rate", "n/a")} -> {enh.get("forbidden_claim_violation_rate", "n/a")}
- Overconfidence rate (baseline -> enhanced): {base.get("overconfidence_rate", "n/a")} -> {enh.get("overconfidence_rate", "n/a")}

## Results
- {_improvement_sentence("Semantic correctness", semantic_base, semantic_enh, lower_is_better=False)}
- {_improvement_sentence("High-risk miss rate", high_risk_miss_base, high_risk_miss_enh, lower_is_better=True)}
- {_improvement_sentence("Citation sufficiency", citation_base, citation_enh, lower_is_better=False)}
- Remaining risks appear in incomplete guidance slices.

## Slice Highlights
- Total slice rows: {len(slice_rows)}
- Review `reports/tables/slice_metrics.csv` for breakdown by `risk_level` and `scenario_type`.

## Limitations
- Uses synthetic/semi-synthetic data and deterministic simulators.
- Not connected to live medical knowledge updates.
- No physician adjudication in this MVP baseline.

## Next Steps
1. Replace simulated inference with real provider-backed LLM calls.
2. Add dual-annotator review with inter-rater agreement.
3. Introduce retrieval quality metrics and abstention calibration curves.
4. Expand high-risk cases and adversarial prompts.

## Safety Disclaimer
This project is for evaluation research only and does not provide medical diagnosis.
"""

    final_path.write_text(final_report, encoding="utf-8")

    top_risks = [
        "Unsafe reassurance in high-risk symptoms",
        "Unsupported claims without evidence",
        "Overconfident tone under weak factual support",
    ]
    guardrails = [
        "High-risk red-flag detector with mandatory emergency escalation",
        "Citation requirement for key claims",
        "Rule-based safety disclaimer insertion",
    ]
    monitoring = [
        "unsafe_advice_rate",
        "high_risk_miss_rate",
        "citation_sufficiency_rate",
        "overconfidence_rate",
        "abstention_appropriateness_rate",
    ]

    unsafe_base = _to_float(base.get("unsafe_advice_rate", "0"))
    unsafe_enh = _to_float(enh.get("unsafe_advice_rate", "0"))
    high_risk_base = _to_float(base.get("high_risk_miss_rate", "0"))
    high_risk_enh = _to_float(enh.get("high_risk_miss_rate", "0"))
    escalation_base = _to_float(base.get("escalation_compliance_rate", "0"))
    escalation_enh = _to_float(enh.get("escalation_compliance_rate", "0"))

    rollback_cfg = eval_cfg.get("deployment", {}).get("rollback", {})
    unsafe_threshold = float(rollback_cfg.get("unsafe_advice_rate_max", 0.20))
    high_risk_threshold = float(rollback_cfg.get("high_risk_miss_rate_max", 0.10))
    consecutive_runs = int(rollback_cfg.get("consecutive_runs", 2))

    onepager = f"""# Deployment Risk & Mitigation One-Pager

## Top Risks
- {top_risks[0]}
- {top_risks[1]}
- {top_risks[2]}

## Guardrails
- {guardrails[0]}
- {guardrails[1]}
- {guardrails[2]}

## Monitoring KPIs
- {monitoring[0]}
- {monitoring[1]}
- {monitoring[2]}
- {monitoring[3]}
- {monitoring[4]}

Current unsafe advice trend (baseline -> enhanced): {_fmt_pct(unsafe_base)} -> {_fmt_pct(unsafe_enh)}
Current high-risk miss trend (baseline -> enhanced): {_fmt_pct(high_risk_base)} -> {_fmt_pct(high_risk_enh)}
Current escalation compliance trend (baseline -> enhanced): {_fmt_pct(escalation_base)} -> {_fmt_pct(escalation_enh)}

## Rollback and Human-in-the-Loop Policy
- Trigger rollback if `unsafe_advice_rate` > {_fmt_pct(unsafe_threshold)} or `high_risk_miss_rate` > {_fmt_pct(high_risk_threshold)} for {consecutive_runs} consecutive runs.
- Route all high-risk/ambiguous cases to human clinician review.
- Log full prompt, retrieved context, and final answer for post-incident audits.
- Maintain a versioned guardrail policy with canary release before broad rollout.

## Safety Position
System must remain advisory and non-diagnostic. High-risk decisions require escalation and human oversight.
"""

    onepager_path.write_text(onepager, encoding="utf-8")
    return str(final_path), str(onepager_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate markdown reports.")
    parser.add_argument("--tables_dir", default="reports/tables")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--eval_csv", default="data/processed/eval_samples.csv")
    parser.add_argument("--eval_config", default="configs/eval.yaml")
    args = parser.parse_args()
    final_path, onepager_path = generate_reports(
        args.tables_dir,
        args.reports_dir,
        args.eval_csv,
        args.eval_config,
    )
    print(f"Wrote: {final_path}")
    print(f"Wrote: {onepager_path}")


if __name__ == "__main__":
    main()
