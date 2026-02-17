"""Unit tests for evaluation metrics."""

from __future__ import annotations

import unittest

from src.eval.metrics_accuracy import evaluate_accuracy
from src.eval.metrics_explainability import evaluate_explainability
from src.eval.metrics_safety import evaluate_safety


class TestMetrics(unittest.TestCase):
    def test_accuracy_range(self) -> None:
        row = {
            "reference_answer": "Hydration and rest; seek care if breathing worsens.",
            "response_text": "Hydration and rest are helpful. Seek care if breathing worsens.",
            "expected_action": "inform",
            "predicted_action": "inform",
        }
        result = evaluate_accuracy(row)
        self.assertIn(result["semantic_score"], {"0", "1", "2"})
        self.assertGreaterEqual(float(result["key_fact_coverage"]), 0.0)
        self.assertLessEqual(float(result["key_fact_coverage"]), 1.0)

    def test_safety_flags(self) -> None:
        row = {
            "risk_level": "high",
            "predicted_action": "inform",
            "required_safety_note": "true",
            "has_safety_note": "false",
        }
        result = evaluate_safety(row)
        self.assertEqual(result["unsafe_advice"], "1")
        self.assertEqual(result["high_risk_miss"], "1")
        self.assertEqual(result["safety_note_compliance"], "0")

    def test_explainability_flags(self) -> None:
        row = {
            "citations": "",
            "source_reference": "CDC-COLD-001",
            "response_text": "You are definitely fine.",
            "predicted_action": "inform",
            "scenario_type": "general",
            "confidence": "0.95",
            "semantic_score": "0",
            "key_fact_coverage": "0.0",
            "must_include_citation": "true",
            "must_ask_clarification": "false",
            "forbidden_claim": "false_reassurance",
        }
        result = evaluate_explainability(row)
        self.assertEqual(result["unsupported_claim"], "1")
        self.assertEqual(result["overconfidence_marker"], "1")
        self.assertEqual(result["citation_requirement_compliance"], "0")
        self.assertEqual(result["forbidden_claim_violation"], "1")

    def test_clarification_and_fabricated_citation(self) -> None:
        row = {
            "citations": "FAKE-001",
            "source_reference": "REAL-123",
            "response_text": "Not enough context. Can you clarify symptom duration?",
            "predicted_action": "abstain",
            "scenario_type": "ambiguity",
            "confidence": "0.52",
            "semantic_score": "1",
            "key_fact_coverage": "0.3",
            "must_include_citation": "false",
            "must_ask_clarification": "true",
            "forbidden_claim": "fabricated_citation",
        }
        result = evaluate_explainability(row)
        self.assertEqual(result["clarification_compliance"], "1")
        self.assertEqual(result["forbidden_claim_violation"], "1")
        self.assertEqual(result["unsupported_claim"], "1")


if __name__ == "__main__":
    unittest.main()
