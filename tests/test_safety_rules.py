"""Unit tests for guardrail safety rules."""

from __future__ import annotations

import unittest

from src.llm.safety_rules import apply_safety_rules


class TestSafetyRules(unittest.TestCase):
    def test_high_risk_escalation(self) -> None:
        out = apply_safety_rules(
            user_query="Chest pain and shortness of breath right now.",
            risk_level="high",
            required_safety_note=True,
            response_text="You should monitor symptoms at home.",
            predicted_action="inform",
        )
        self.assertEqual(out["predicted_action"], "emergency_escalation")
        self.assertIn("emergency", out["response_text"].lower())
        self.assertEqual(out["has_safety_note"], "true")


if __name__ == "__main__":
    unittest.main()

