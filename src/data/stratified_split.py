"""Stratified split helper for eval datasets."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

from src.utils.io import read_csv, write_csv

DEDUP_FIELDS = [
    "scenario_type",
    "risk_level",
    "expected_action",
    "source_reference",
    "reference_answer",
    "forbidden_claim",
    "red_flag_tags",
]


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _dedup_group_key(row: dict[str, str]) -> str:
    parts = [_normalize_text(row.get(field, "")) for field in DEDUP_FIELDS]
    return "||".join(parts)


def assign_splits(
    rows: list[dict[str, str]],
    test_ratio: float = 0.3,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Assign `dev/test` split stratified by scenario and risk."""
    groups: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = f"{row.get('scenario_type', '')}|{row.get('risk_level', '')}"
        dedup_key = _dedup_group_key(row)
        groups[key][dedup_key].append(row)

    rng = random.Random(seed)
    for group_map in groups.values():
        grouped = list(group_map.values())
        rng.shuffle(grouped)
        total_rows = sum(len(bucket) for bucket in grouped)
        target_test = max(1, int(round(total_rows * test_ratio))) if total_rows > 1 else 0

        test_buckets: list[list[dict[str, str]]] = []
        dev_buckets: list[list[dict[str, str]]] = []
        current_test = 0

        for bucket in grouped:
            size = len(bucket)
            add_to_test = False
            if target_test > 0:
                dist_if_test = abs((current_test + size) - target_test)
                dist_if_dev = abs(current_test - target_test)
                add_to_test = dist_if_test <= dist_if_dev and current_test < target_test

            if add_to_test:
                test_buckets.append(bucket)
                current_test += size
            else:
                dev_buckets.append(bucket)

        if target_test > 0 and not test_buckets and dev_buckets:
            smallest = min(dev_buckets, key=len)
            dev_buckets.remove(smallest)
            test_buckets.append(smallest)

        if total_rows > 1 and not dev_buckets and len(test_buckets) > 1:
            smallest = min(test_buckets, key=len)
            test_buckets.remove(smallest)
            dev_buckets.append(smallest)

        for bucket in test_buckets:
            for row in bucket:
                row["split"] = "test"
        for bucket in dev_buckets:
            for row in bucket:
                row["split"] = "dev"

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
