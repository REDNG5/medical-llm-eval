"""Tests for RAG external knowledge base and decoding alignment."""

from __future__ import annotations

import unittest

from src.llm.rag_infer import load_knowledge_base, retrieve_context
from src.utils.io import load_yaml


class TestRagSetup(unittest.TestCase):
    def test_external_knowledge_base_loads(self) -> None:
        kb = load_knowledge_base("data/raw/knowledge_base.json")
        self.assertGreaterEqual(len(kb), 5)
        first = kb[0]
        self.assertIn("id", first)
        self.assertIn("snippet", first)
        self.assertIn("keywords", first)

    def test_retrieve_respects_top_k(self) -> None:
        kb = load_knowledge_base("data/raw/knowledge_base.json")
        hits = retrieve_context("I have chest pain and shortness of breath", kb, top_k=1)
        self.assertLessEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "AHA-CHEST-911")

    def test_decoding_config_is_aligned(self) -> None:
        cfg = load_yaml("configs/models.yaml")
        shared = cfg.get("shared_decoding", {})
        baseline = {**shared, **cfg.get("baseline", {})}
        enhanced = {**shared, **cfg.get("enhanced", {})}
        for key in ["llm_mode", "provider", "model_name", "temperature", "top_p", "max_tokens"]:
            self.assertEqual(baseline.get(key), enhanced.get(key), msg=f"Mismatch at {key}")


if __name__ == "__main__":
    unittest.main()
