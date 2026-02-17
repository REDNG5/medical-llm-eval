"""Unit tests for eval template schema."""

from __future__ import annotations

import csv
from pathlib import Path
import unittest


EXPECTED_COLUMNS = [
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


class TestDataSchema(unittest.TestCase):
    def test_template_columns(self) -> None:
        path = Path("data/eval_set/eval_samples_template.csv")
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        self.assertEqual(header, EXPECTED_COLUMNS)


if __name__ == "__main__":
    unittest.main()
