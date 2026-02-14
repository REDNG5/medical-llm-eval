"""Error distribution analysis by taxonomy tag."""

from __future__ import annotations

import argparse
from collections import Counter

from src.utils.io import read_csv, write_csv


def run_error_analysis(per_sample_csv: str, output_csv: str) -> None:
    """Aggregate error tags into counts."""
    rows = read_csv(per_sample_csv)
    counter: Counter[str] = Counter()
    for row in rows:
        for tag in [t for t in row.get("error_tags", "").split(";") if t]:
            counter[tag] += 1

    out_rows = [{"error_type": k, "count": str(v)} for k, v in counter.items()]
    out_rows.sort(key=lambda r: int(r["count"]), reverse=True)
    write_csv(output_csv, out_rows, ["error_type", "count"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Error analysis by taxonomy.")
    parser.add_argument("--per_sample_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    run_error_analysis(args.per_sample_csv, args.output_csv)


if __name__ == "__main__":
    main()

