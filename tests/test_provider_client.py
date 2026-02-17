"""Unit tests for provider client structured parsing and fallback."""

from __future__ import annotations

import unittest

from src.llm.provider_client import _parse_structured_output, generate_structured_with_fallback


class TestProviderClient(unittest.TestCase):
    def test_parse_structured_output(self) -> None:
        raw = (
            "ACTION: advise_visit\n"
            "CONFIDENCE: 0.78\n"
            "CITATIONS: SRC-1;SRC-2\n"
            "ANSWER: Please arrange a clinician visit soon."
        )
        out = _parse_structured_output(
            raw,
            default_action="inform",
            default_answer="fallback",
            default_citations=[],
            default_confidence=0.5,
        )
        self.assertEqual(out.predicted_action, "advise_visit")
        self.assertAlmostEqual(out.confidence, 0.78, places=2)
        self.assertEqual(out.citations, ["SRC-1", "SRC-2"])
        self.assertIn("clinician visit", out.response_text)
        self.assertEqual(out.generation_source, "openai")
        self.assertEqual(out.fallback_reason, "none")
        self.assertTrue(out.api_success)

    def test_mock_fallback(self) -> None:
        out = generate_structured_with_fallback(
            prompt="Test prompt",
            model_cfg={"llm_mode": "mock"},
            default_action="inform",
            default_answer="Fallback answer.",
            default_citations=["SRC-3"],
            default_confidence=0.61,
        )
        self.assertEqual(out.predicted_action, "inform")
        self.assertEqual(out.response_text, "Fallback answer.")
        self.assertEqual(out.citations, ["SRC-3"])
        self.assertAlmostEqual(out.confidence, 0.61, places=2)
        self.assertEqual(out.generation_source, "mock")
        self.assertEqual(out.fallback_reason, "none")
        self.assertFalse(out.api_success)


if __name__ == "__main__":
    unittest.main()
