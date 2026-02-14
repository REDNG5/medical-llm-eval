"""Stratified split helper for eval datasets."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

from src.utils.io import read_csv, write_csv


def assign_splits(
    rows: list[dict[str, str]],
    test_ratio: float = 0.3,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Assign `dev/test` split stratified by scenario and risk."""
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('scenario_type', '')}|{row.get('risk_level', '')}"
        groups[key].append(row)

    rng = random.Random(seed)
    for grouped in groups.values():
        rng.shuffle(grouped)
        n_test = max(1, int(len(grouped) * test_ratio)) if len(grouped) > 1 else 0
        for idx, row in enumerate(grouped):
            row["split"] = "test" if idx < n_test else "dev"

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified split for eval CSV.")
    parser.add_argument("--input_csv", default="data/processed/eval_samples.csv")
    parser.add_argument("--output_csv", default="data/processed/eval_samples.csv")
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_csv(args.input_csv)
    rows = assign_splits(rows, test_ratio=args.test_ratio, seed=args.seed)
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(args.output_csv, rows, fieldnames)
    print(f"Wrote stratified splits to {args.output_csv}")


if __name__ == "__main__":
    main()

