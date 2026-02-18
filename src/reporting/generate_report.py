"""Generate final markdown reports from computed metrics."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.eval.error_taxonomy import TAXONOMY
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


def _truncate(text: str, max_len: int = 360) -> str:
    clean = " ".join((text or "").strip().split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3].rstrip() + "..."


def _parse_tags(raw: str) -> list[str]:
    return [t.strip() for t in (raw or "").split(";") if t.strip()]


def _tag_severity(tag: str) -> int:
    weights = {
        "missing_red_flag": 5,
        "unsafe_reassurance": 5,
        "factual_error": 4,
        "unsupported_claim": 4,
        "overconfident_uncertain": 3,
        "incomplete_guidance": 2,
    }
    return weights.get(tag, 1)


def _choose_primary_tag(enhanced_tags: list[str], baseline_tags: list[str]) -> str:
    candidates = enhanced_tags or baseline_tags
    if not candidates:
        return "incomplete_guidance"
    return sorted(candidates, key=_tag_severity, reverse=True)[0]


def _build_failure_case_section(
    *,
    tables_dir: str,
    eval_rows: list[dict[str, str]],
    eval_split: str,
    max_cases: int = 5,
    min_cases: int = 3,
) -> str:
    suffix = eval_split if eval_split in {"dev", "test", "all"} else "test"
    baseline_path = Path(tables_dir) / f"per_sample_eval_baseline_{suffix}.csv"
    enhanced_path = Path(tables_dir) / f"per_sample_eval_enhanced_{suffix}.csv"
    if not baseline_path.exists() or not enhanced_path.exists():
        return (
            "## Failure Case Cards\n"
            "- Not available: missing per-sample evaluation files for baseline/enhanced.\n"
        )

    baseline_rows = read_csv(baseline_path)
    enhanced_rows = read_csv(enhanced_path)
    base_by_id = {row.get("sample_id", ""): row for row in baseline_rows}
    enh_by_id = {row.get("sample_id", ""): row for row in enhanced_rows}
    eval_by_id = {row.get("sample_id", ""): row for row in eval_rows}

    candidates: list[tuple[int, str]] = []
    for sample_id, enh in enh_by_id.items():
        base = base_by_id.get(sample_id)
        ref = eval_by_id.get(sample_id, {})
        if not base or not ref:
            continue
        enh_tags = _parse_tags(enh.get("error_tags", ""))
        base_tags = _parse_tags(base.get("error_tags", ""))
        if not enh_tags and not base_tags:
            continue

        persistent = len(set(enh_tags) & set(base_tags))
        score = sum(_tag_severity(tag) for tag in enh_tags) + persistent
        if enh.get("high_risk_miss", "0") == "1" or enh.get("unsafe_advice", "0") == "1":
            score += 3
        if enh.get("semantic_score", "0") == "0":
            score += 2
        candidates.append((score, sample_id))

    candidates.sort(key=lambda x: (-x[0], x[1]))
    chosen_ids = [sample_id for _, sample_id in candidates[:max_cases]]

    if len(chosen_ids) < min_cases:
        # Backfill with baseline-only failures to keep 3-5 cards.
        for _, sample_id in candidates[max_cases:]:
            if sample_id not in chosen_ids:
                chosen_ids.append(sample_id)
            if len(chosen_ids) >= min_cases:
                break

    if not chosen_ids:
        return "## Failure Case Cards\n- No failure cases detected on this split.\n"

    lines = ["## Failure Case Cards", ""]
    for idx, sample_id in enumerate(chosen_ids, start=1):
        base = base_by_id[sample_id]
        enh = enh_by_id[sample_id]
        ref = eval_by_id.get(sample_id, {})
        base_tags = _parse_tags(base.get("error_tags", ""))
        enh_tags = _parse_tags(enh.get("error_tags", ""))
        primary_tag = _choose_primary_tag(enh_tags, base_tags)
        taxonomy = TAXONOMY.get(primary_tag, {})
        root_cause = taxonomy.get("likely_cause", "Multi-factor prompt/retrieval/policy mismatch.")
        remediation = taxonomy.get("remediation_action", "Refine prompts and rules; add targeted tests.")

        lines.append(f"### Case {idx} ({sample_id})")
        lines.append(f"- Query: {_truncate(ref.get('user_query', ''))}")
        lines.append(f"- Reference: {_truncate(ref.get('reference_answer', ''))}")
        lines.append(f"- Baseline Output: {_truncate(base.get('response_text', ''))}")
        lines.append(f"- Enhanced Output: {_truncate(enh.get('response_text', ''))}")
        lines.append(f"- Baseline Error Tags: {', '.join(base_tags) if base_tags else 'none'}")
        lines.append(f"- Enhanced Error Tags: {', '.join(enh_tags) if enh_tags else 'none'}")
        lines.append(f"- Root Cause: {_truncate(root_cause, max_len=220)}")
        lines.append(f"- Fix Recommendation: {_truncate(remediation, max_len=220)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


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
    failure_case_section = _build_failure_case_section(
        tables_dir=tables_dir,
        eval_rows=eval_rows,
        eval_split=eval_split,
    )

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

{failure_case_section}

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
