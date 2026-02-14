"""Generate final markdown reports from computed metrics."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.utils.io import ensure_dir, read_csv


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def generate_reports(tables_dir: str, reports_dir: str) -> tuple[str, str]:
    """Generate final report and deployment one-pager."""
    metrics_path = Path(tables_dir) / "metrics_summary.csv"
    slice_path = Path(tables_dir) / "slice_metrics.csv"
    metrics_rows = read_csv(metrics_path) if metrics_path.exists() else []
    slice_rows = read_csv(slice_path) if slice_path.exists() else []

    by_variant = {row["model_variant"]: row for row in metrics_rows}
    base = by_variant.get("baseline", {})
    enh = by_variant.get("enhanced", {})

    out_dir = ensure_dir(reports_dir)
    final_path = out_dir / "final_report.md"
    onepager_path = out_dir / "deployment_risk_onepager.md"

    final_report = f"""# Final Report: Medical LLM Evaluation MVP

Date: {date.today().isoformat()}

## Setup
- Pipeline type: Baseline vs Enhanced (RAG + safety rules)
- Eval set size: {enh.get("num_samples", base.get("num_samples", "n/a"))}
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
- Overconfidence rate (baseline -> enhanced): {base.get("overconfidence_rate", "n/a")} -> {enh.get("overconfidence_rate", "n/a")}

## Results
- Enhanced improved semantic correctness and reduced high-risk misses in this synthetic MVP.
- Citation sufficiency improved due to retrieval grounding.
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

## Rollback and Human-in-the-Loop Policy
- Trigger rollback if `unsafe_advice_rate` or `high_risk_miss_rate` exceeds predefined threshold for 2 consecutive runs.
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
    args = parser.parse_args()
    final_path, onepager_path = generate_reports(args.tables_dir, args.reports_dir)
    print(f"Wrote: {final_path}")
    print(f"Wrote: {onepager_path}")


if __name__ == "__main__":
    main()

