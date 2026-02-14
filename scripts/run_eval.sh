#!/usr/bin/env bash
set -euo pipefail

python -m src.eval.evaluator --eval_csv data/processed/eval_samples.csv --pred_csv reports/tables/predictions_baseline.csv --output_dir reports/tables
python -m src.eval.evaluator --eval_csv data/processed/eval_samples.csv --pred_csv reports/tables/predictions_enhanced.csv --output_dir reports/tables

python -m src.analysis.slice_analysis --per_sample_csv reports/tables/per_sample_eval_baseline.csv --output_csv reports/tables/slice_metrics_baseline.csv
python -m src.analysis.slice_analysis --per_sample_csv reports/tables/per_sample_eval_enhanced.csv --output_csv reports/tables/slice_metrics_enhanced.csv

python -m src.analysis.error_analysis --per_sample_csv reports/tables/per_sample_eval_baseline.csv --output_csv reports/tables/error_counts_baseline.csv
python -m src.analysis.error_analysis --per_sample_csv reports/tables/per_sample_eval_enhanced.csv --output_csv reports/tables/error_counts_enhanced.csv

