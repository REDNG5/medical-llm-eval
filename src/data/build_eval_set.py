"""Build a semi-automatic medical eval set from template seeds."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.data.stratified_split import assign_splits
from src.utils.io import load_yaml, parse_bool, read_csv, write_csv
from src.utils.logging import get_logger
from src.utils.seed import set_seed

REQUIRED_COLUMNS = [
    "sample_id",
    "user_query",
    "scenario_type",
    "risk_level",
    "reference_answer",
    "required_safety_note",
    "expected_action",
    "source_reference",
    "must_ask_clarification",
    "forbidden_claim",
    "must_include_citation",
    "red_flag_tags",
    "split",
]

QUERY_PREFIXES = [
    "What should I do if",
    "I am worried because",
    "Can you help me understand",
    "Is it serious if",
    "How should I handle",
]


def _base_seed_rows(template_csv: str) -> list[dict[str, str]]:
    rows = read_csv(template_csv)
    for row in rows:
        for col in REQUIRED_COLUMNS:
            row.setdefault(col, "")
    return rows


def _next_augmented_index(rows: list[dict[str, str]]) -> int:
    max_idx = 0
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if sample_id.startswith("S") and sample_id[1:].isdigit():
            max_idx = max(max_idx, int(sample_id[1:]))
    return max_idx + 1


def _augment_row(row: dict[str, str], new_id: str, rng: random.Random) -> dict[str, str]:
    query = row["user_query"].rstrip(" ?.")
    prefix = rng.choice(QUERY_PREFIXES)
    variant_query = f"{prefix.lower()} {query.lower()}?"

    new_row = dict(row)
    new_row["sample_id"] = new_id
    new_row["user_query"] = variant_query[0].upper() + variant_query[1:]
    new_row["required_safety_note"] = str(parse_bool(row.get("required_safety_note", False))).lower()
    new_row["must_ask_clarification"] = str(parse_bool(row.get("must_ask_clarification", False))).lower()
    new_row["must_include_citation"] = str(parse_bool(row.get("must_include_citation", False))).lower()
    new_row["split"] = ""
    return new_row


def build_eval_set(
    template_csv: str,
    output_csv: str,
    target_samples: int,
    test_ratio: float,
    seed: int,
) -> list[dict[str, str]]:
    """Build target-size eval set by expanding template rows and stratifying split."""
    set_seed(seed)
    rng = random.Random(seed)
    rows = _base_seed_rows(template_csv)
    if not rows:
        raise ValueError("Template CSV is empty.")
    seed_rows = list(rows)
    existing_ids = {str(row.get("sample_id", "")) for row in rows}

    current_count = len(rows)
    next_idx = _next_augmented_index(rows)
    while current_count < target_samples:
        source = seed_rows[(current_count - len(seed_rows)) % len(seed_rows)]
        sample_id = f"S{next_idx:04d}"
        while sample_id in existing_ids:
            next_idx += 1
            sample_id = f"S{next_idx:04d}"
        rows.append(_augment_row(source, sample_id, rng))
        existing_ids.add(sample_id)
        current_count += 1
        next_idx += 1

    rows = assign_splits(rows, test_ratio=test_ratio, seed=seed)
    write_csv(output_csv, rows, REQUIRED_COLUMNS)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluation set.")
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--eval_config", default="configs/eval.yaml")
    args = parser.parse_args()

    base_cfg = load_yaml(args.base_config)
    eval_cfg = load_yaml(args.eval_config)
    seed = int(base_cfg.get("project", {}).get("seed", 42))
    template_csv = str(base_cfg.get("paths", {}).get("eval_template_csv", "data/eval_set/eval_samples_template.csv"))
    output_csv = str(base_cfg.get("paths", {}).get("eval_processed_csv", "data/processed/eval_samples.csv"))
    target_samples = int(eval_cfg.get("dataset", {}).get("target_samples", 120))
    test_ratio = float(eval_cfg.get("dataset", {}).get("test_ratio", 0.3))

    logger = get_logger("build_eval_set")
    rows = build_eval_set(
        template_csv=template_csv,
        output_csv=output_csv,
        target_samples=target_samples,
        test_ratio=test_ratio,
        seed=seed,
    )
    logger.info("Built eval set with %s samples at %s", len(rows), Path(output_csv))


if __name__ == "__main__":
    main()
