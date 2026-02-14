"""Text preprocessing for eval queries and references."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.utils.io import read_csv, write_csv


def normalize_text(text: str) -> str:
    """Normalize whitespace and punctuation spacing."""
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = cleaned.replace(" ,", ",").replace(" .", ".")
    return cleaned


def preprocess_eval_csv(input_csv: str, output_csv: str) -> None:
    """Normalize text fields in evaluation CSV."""
    rows = read_csv(input_csv)
    for row in rows:
        row["user_query"] = normalize_text(row.get("user_query", ""))
        row["reference_answer"] = normalize_text(row.get("reference_answer", ""))
    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(output_csv, rows, fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess eval CSV text fields.")
    parser.add_argument("--input_csv", default="data/eval_set/eval_samples_template.csv")
    parser.add_argument("--output_csv", default="data/processed/eval_samples_clean.csv")
    args = parser.parse_args()
    preprocess_eval_csv(args.input_csv, args.output_csv)
    print(f"Wrote preprocessed file: {Path(args.output_csv)}")


if __name__ == "__main__":
    main()

