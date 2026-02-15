#!/usr/bin/env bash
set -euo pipefail

python -m src.eval.evaluator --eval_csv data/processed/eval_samples.csv --pred_csv reports/tables/predictions_baseline.csv --output_dir reports/tables --split test --eval_config configs/eval.yaml
python -m src.eval.evaluator --eval_csv data/processed/eval_samples.csv --pred_csv reports/tables/predictions_enhanced.csv --output_dir reports/tables --split test --eval_config configs/eval.yaml

python -m src.analysis.slice_analysis --per_sample_csv reports/tables/per_sample_eval_baseline_test.csv --output_csv reports/tables/slice_metrics_baseline_test.csv
python -m src.analysis.slice_analysis --per_sample_csv reports/tables/per_sample_eval_enhanced_test.csv --output_csv reports/tables/slice_metrics_enhanced_test.csv

python -m src.analysis.error_analysis --per_sample_csv reports/tables/per_sample_eval_baseline_test.csv --output_csv reports/tables/error_counts_baseline_test.csv
python -m src.analysis.error_analysis --per_sample_csv reports/tables/per_sample_eval_enhanced_test.csv --output_csv reports/tables/error_counts_enhanced_test.csv
