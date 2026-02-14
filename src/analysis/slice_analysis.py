"""Slice metrics by risk level and scenario type."""

from __future__ import annotations

import argparse
from collections import defaultdict

from src.utils.io import read_csv, write_csv


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def run_slice_analysis(per_sample_csv: str, output_csv: str) -> None:
    """Compute per-slice rates and save to CSV."""
    rows = read_csv(per_sample_csv)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("risk_level", ""), row.get("scenario_type", ""))
        groups[key].append(row)

    out_rows: list[dict[str, str]] = []
    for (risk, scenario), bucket in groups.items():
        n = len(bucket)
        if n == 0:
            continue
        semantic_mean = sum(_to_int(r.get("semantic_score", "0")) for r in bucket) / n
        unsafe_rate = sum(_to_int(r.get("unsafe_advice", "0")) for r in bucket) / n
        cite_rate = sum(_to_int(r.get("citation_sufficiency", "0")) for r in bucket) / n
        overconf_rate = sum(_to_int(r.get("overconfidence_marker", "0")) for r in bucket) / n
        variant = bucket[0].get("model_variant", "unknown")

        out_rows.append(
            {
                "model_variant": variant,
                "risk_level": risk,
                "scenario_type": scenario,
                "num_samples": str(n),
                "semantic_score_mean": f"{semantic_mean:.4f}",
                "unsafe_advice_rate": f"{unsafe_rate:.4f}",
                "citation_sufficiency_rate": f"{cite_rate:.4f}",
                "overconfidence_rate": f"{overconf_rate:.4f}",
            }
        )

    out_rows.sort(key=lambda r: (r["model_variant"], r["risk_level"], r["scenario_type"]))
    write_csv(
        output_csv,
        out_rows,
        [
            "model_variant",
            "risk_level",
            "scenario_type",
            "num_samples",
            "semantic_score_mean",
            "unsafe_advice_rate",
            "citation_sufficiency_rate",
            "overconfidence_rate",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice analysis.")
    parser.add_argument("--per_sample_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    run_slice_analysis(args.per_sample_csv, args.output_csv)


if __name__ == "__main__":
    main()

