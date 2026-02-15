"""Regression tests to ensure inference does not use label columns."""

from __future__ import annotations

import unittest

from src.llm.baseline_infer import _predict_action_baseline
from src.llm.inference_common import sanitize_infer_input
from src.llm.rag_infer import _predict_action


class TestNoLabelLeakage(unittest.TestCase):
    def test_sanitize_infer_input_drops_labels(self) -> None:
        row = {
            "sample_id": "S1",
            "user_query": "I have chest pain.",
            "risk_level": "low",
            "scenario_type": "general",
            "expected_action": "inform",
        }
        infer_row = sanitize_infer_input(row)
        self.assertEqual(set(infer_row.keys()), {"sample_id", "user_query"})

    def test_baseline_action_ignores_label_columns(self) -> None:
        row_a = {
            "sample_id": "S1",
            "user_query": "I have chest pain and shortness of breath.",
            "risk_level": "low",
            "scenario_type": "general",
            "expected_action": "inform",
        }
        row_b = {
            "sample_id": "S1",
            "user_query": "I have chest pain and shortness of breath.",
            "risk_level": "high",
            "scenario_type": "high_risk",
            "expected_action": "emergency_escalation",
        }
        action_a = _predict_action_baseline(sanitize_infer_input(row_a))
        action_b = _predict_action_baseline(sanitize_infer_input(row_b))
        self.assertEqual(action_a, action_b)

    def test_enhanced_action_ignores_label_columns(self) -> None:
        row_a = {
            "sample_id": "S2",
            "user_query": "Not sure what is wrong, mild rash maybe.",
            "risk_level": "low",
            "scenario_type": "general",
            "expected_action": "inform",
        }
        row_b = {
            "sample_id": "S2",
            "user_query": "Not sure what is wrong, mild rash maybe.",
            "risk_level": "high",
            "scenario_type": "high_risk",
            "expected_action": "emergency_escalation",
        }
        action_a = _predict_action(sanitize_infer_input(row_a), retrieved=[])
        action_b = _predict_action(sanitize_infer_input(row_b), retrieved=[])
        self.assertEqual(action_a, action_b)


if __name__ == "__main__":
    unittest.main()

