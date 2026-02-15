"""Combine baseline/enhanced metrics into report tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.io import ensure_dir, read_csv, write_csv


def make_tables(tables_dir: str) -> tuple[str, str]:
    """Build `metrics_summary.csv` and `slice_metrics.csv`."""
    tdir = ensure_dir(tables_dir)

    metrics_rows: list[dict[str, str]] = []
    metric_candidates = [
        ["metrics_baseline_test.csv", "metrics_baseline.csv"],
        ["metrics_enhanced_test.csv", "metrics_enhanced.csv"],
    ]
    for candidates in metric_candidates:
        for file_name in candidates:
            path = tdir / file_name
            if path.exists():
                metrics_rows.extend(read_csv(path))
                break

    metrics_summary_path = str(tdir / "metrics_summary.csv")
    if metrics_rows:
        metric_fields = list(metrics_rows[0].keys())
        write_csv(metrics_summary_path, metrics_rows, metric_fields)

    slice_rows: list[dict[str, str]] = []
    slice_candidates = [
        ["slice_metrics_baseline_test.csv", "slice_metrics_baseline.csv"],
        ["slice_metrics_enhanced_test.csv", "slice_metrics_enhanced.csv"],
    ]
    for candidates in slice_candidates:
        for file_name in candidates:
            path = tdir / file_name
            if path.exists():
                slice_rows.extend(read_csv(path))
                break

    slice_summary_path = str(tdir / "slice_metrics.csv")
    if slice_rows:
        slice_fields = list(slice_rows[0].keys())
        write_csv(slice_summary_path, slice_rows, slice_fields)

    return metrics_summary_path, slice_summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate summary tables.")
    parser.add_argument("--tables_dir", default="reports/tables")
    args = parser.parse_args()
    metrics_path, slice_path = make_tables(args.tables_dir)
    print(f"Wrote: {Path(metrics_path)}")
    print(f"Wrote: {Path(slice_path)}")


if __name__ == "__main__":
    main()
