#!/usr/bin/env bash
set -euo pipefail

python -m src.llm.rag_infer \
  --input_csv data/processed/eval_samples.csv \
  --output_csv reports/tables/predictions_enhanced.csv \
  --metadata_out reports/tables/run_metadata_enhanced.json

