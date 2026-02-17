# Medical LLM Eval: Evaluation and Hallucination Analysis MVP

## Project Brief

### Problem statement
We need to evaluate whether an LLM can provide useful and safe responses in a medical information setting (not diagnosis).

### Goal
Build an evaluation pipeline that compares:
- (A) baseline LLM (no retrieval)
- (B) enhanced system (RAG + rule-based safety layer)

### Scope
- 100-300 evaluation samples, semi-automatically constructed + manually reviewed.

### Key outputs
1. Accuracy metrics
2. Safety metrics
3. Explainability / citation sufficiency metrics
4. Error taxonomy and root-cause analysis
5. One-page deployment risk & mitigation strategy

### Constraints
- Python-first
- Reproducible experiments
- Clear separation of data, prompts, models, evaluation, reports
- No medical diagnosis claims; include safety disclaimer

## Safety Notice
This project is for model evaluation research only. It does not provide medical diagnosis. All generated responses must include or preserve a non-diagnostic safety disclaimer for medical advice workflows.

## Architecture
The repository separates concerns into:
- `data/`: raw inputs, processed sets, eval templates
- `src/llm/`: baseline and enhanced inference logic
- `src/eval/`: metric calculations + error taxonomy
- `src/analysis/`: slice and error analysis
- `src/reporting/`: report tables/figures/markdown generation
- `reports/`: generated artifacts

## Quickstart

```bash
cd medical-llm-eval
make build_eval
make run_baseline
make run_enhanced
make evaluate
make report
```

Or full pipeline:

```bash
make run_all
```

## Real LLM Mode (Optional)
- Default mode is `mock` (no external API call).
- To enable OpenAI-backed generation:

```bash
pip install -e ".[llm]"
# bash
export OPENAI_API_KEY=your_key
export LLM_MODE=openai
# powershell
$env:OPENAI_API_KEY="your_key"
$env:LLM_MODE="openai"
```

Then run the same pipeline commands (`make run_all` or module commands). If API/key is unavailable, the code falls back to mock behavior.

## Reproducibility
- Seeded generation and split logic (`src/utils/seed.py`)
- Logged prompts and decoding config in `reports/tables/run_metadata_*.json`
- Metadata also logs `generation_source`, `fallback_reason`, and `api_success_count`
- Shared decoding parameters for baseline/enhanced in `configs/models.yaml` (`shared_decoding`)
- External RAG retrieval corpus in `data/raw/knowledge_base.json` (no in-code hardcoded KB)
- Config-driven runs via `configs/*.yaml`
- Evaluation defaults to `test` split (`src.eval.evaluator --split test`)
- Dedup-aware split assignment prevents near-duplicate template variants crossing `dev/test`

## Evaluation Design
- **Accuracy**: semantic correctness rubric (0-2), key fact coverage
- **Safety**: unsafe advice rate, high-risk miss rate, escalation compliance
- **Explainability**: citation sufficiency, citation requirement compliance, unsupported claim rate
- **Calibration/behavior**: overconfidence marker, abstention appropriateness
- **Policy adherence**: forbidden-claim violation rate, clarification compliance

## Expected Artifacts
- `reports/tables/metrics_summary.csv`
- `reports/tables/slice_metrics.csv`
- `reports/figures/error_distribution.png`
- `reports/figures/baseline_vs_enhanced.png`
- `reports/final_report.md`
- `reports/deployment_risk_onepager.md`
