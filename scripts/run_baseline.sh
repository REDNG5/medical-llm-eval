#!/usr/bin/env bash
set -euo pipefail

python -m src.llm.baseline_infer \
  --input_csv data/processed/eval_samples.csv \
  --output_csv reports/tables/predictions_baseline.csv \
  --metadata_out reports/tables/run_metadata_baseline.json

