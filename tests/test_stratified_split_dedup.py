"""Tests for split assignment with dedup grouping."""

from __future__ import annotations

import unittest

from src.data.stratified_split import assign_splits


class TestStratifiedSplitDedup(unittest.TestCase):
    def test_near_duplicates_stay_in_same_split(self) -> None:
        rows = [
            {
                "sample_id": "T1",
                "user_query": "Chest pain with shortness of breath.",
                "scenario_type": "high_risk",
                "risk_level": "high",
                "expected_action": "emergency_escalation",
                "source_reference": "AHA-CHEST-911",
                "reference_answer": "Seek emergency care now.",
                "forbidden_claim": "ignore_emergency",
                "red_flag_tags": "chest_pain;dyspnea",
            },
            {
                "sample_id": "S1",
                "user_query": "What should I do if chest pain with shortness of breath?",
                "scenario_type": "high_risk",
                "risk_level": "high",
                "expected_action": "emergency_escalation",
                "source_reference": "AHA-CHEST-911",
                "reference_answer": "Seek emergency care now.",
                "forbidden_claim": "ignore_emergency",
                "red_flag_tags": "chest_pain;dyspnea",
            },
            {
                "sample_id": "T2",
                "user_query": "Mild sore throat for two days.",
                "scenario_type": "general",
                "risk_level": "low",
                "expected_action": "inform",
                "source_reference": "CDC-COLD-001",
                "reference_answer": "Hydrate and rest.",
                "forbidden_claim": "definitive_diagnosis",
                "red_flag_tags": "none",
            },
        ]
        out = assign_splits(rows, test_ratio=0.5, seed=42)
        split_map = {row["sample_id"]: row["split"] for row in out}
        self.assertEqual(split_map["T1"], split_map["S1"])


if __name__ == "__main__":
    unittest.main()
