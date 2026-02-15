PYTHON ?= python

.PHONY: setup setup_llm build_eval run_baseline run_enhanced evaluate report run_all test

setup:
	$(PYTHON) -m pip install -e ".[dev,viz]"

setup_llm:
	$(PYTHON) -m pip install -e ".[dev,viz,llm]"

build_eval:
	$(PYTHON) -m src.data.build_eval_set --base_config configs/base.yaml --eval_config configs/eval.yaml

run_baseline:
	$(PYTHON) -m src.llm.baseline_infer --input_csv data/processed/eval_samples.csv --output_csv reports/tables/predictions_baseline.csv --metadata_out reports/tables/run_metadata_baseline.json

run_enhanced:
	$(PYTHON) -m src.llm.rag_infer --input_csv data/processed/eval_samples.csv --output_csv reports/tables/predictions_enhanced.csv --metadata_out reports/tables/run_metadata_enhanced.json

evaluate:
	$(PYTHON) -m src.eval.evaluator --eval_csv data/processed/eval_samples.csv --pred_csv reports/tables/predictions_baseline.csv --output_dir reports/tables --split test --eval_config configs/eval.yaml
	$(PYTHON) -m src.eval.evaluator --eval_csv data/processed/eval_samples.csv --pred_csv reports/tables/predictions_enhanced.csv --output_dir reports/tables --split test --eval_config configs/eval.yaml
	$(PYTHON) -m src.analysis.slice_analysis --per_sample_csv reports/tables/per_sample_eval_baseline_test.csv --output_csv reports/tables/slice_metrics_baseline_test.csv
	$(PYTHON) -m src.analysis.slice_analysis --per_sample_csv reports/tables/per_sample_eval_enhanced_test.csv --output_csv reports/tables/slice_metrics_enhanced_test.csv
	$(PYTHON) -m src.analysis.error_analysis --per_sample_csv reports/tables/per_sample_eval_baseline_test.csv --output_csv reports/tables/error_counts_baseline_test.csv
	$(PYTHON) -m src.analysis.error_analysis --per_sample_csv reports/tables/per_sample_eval_enhanced_test.csv --output_csv reports/tables/error_counts_enhanced_test.csv

report:
	$(PYTHON) -m src.reporting.make_tables --tables_dir reports/tables
	$(PYTHON) -m src.reporting.make_figures --tables_dir reports/tables --figures_dir reports/figures
	$(PYTHON) -m src.reporting.generate_report --tables_dir reports/tables --reports_dir reports

run_all: build_eval run_baseline run_enhanced evaluate report

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"
