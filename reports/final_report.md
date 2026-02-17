# Final Report: Medical LLM Evaluation MVP

Date: 2026-02-17

## Setup
- Pipeline type: Baseline vs Enhanced (RAG + safety rules)
- Eval set size: total=120, test=42
- Task scope: medical information and symptom triage support (non-diagnostic)

## Methodology
- Data: semi-automatic evaluation set with manual-review-ready template.
- Baseline: direct answer without retrieval.
- Enhanced: retrieval-augmented drafting plus deterministic safety guardrails.
- Metrics: accuracy, safety, explainability, calibration behavior.

## Metrics
- Semantic score mean (baseline -> enhanced): 0.7619 -> 1.0952
- Unsafe advice rate (baseline -> enhanced): 0.1429 -> 0.0476
- Citation sufficiency rate (baseline -> enhanced): 0.0000 -> 0.4524
- Citation requirement compliance (baseline -> enhanced): 0.1905 -> 0.5714
- Forbidden-claim violation rate (baseline -> enhanced): 0.1429 -> 0.0476
- Overconfidence rate (baseline -> enhanced): 0.5476 -> 0.3095

## Results
- Semantic correctness improved (0.7619 -> 1.0952).
- High-risk miss rate improved (0.1429 -> 0.0476).
- Citation sufficiency improved (0.0000 -> 0.4524).
- Remaining risks appear in incomplete guidance slices.

## Slice Highlights
- Total slice rows: 12
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
