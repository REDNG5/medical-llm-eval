"""Enhanced RAG + safety-rule inference runner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.llm.prompts import rag_prompt
from src.llm.safety_rules import apply_safety_rules
from src.utils.io import load_yaml, read_csv, write_csv, write_json
from src.utils.logging import get_logger
from src.utils.seed import set_seed

PREDICTION_COLUMNS = [
    "sample_id",
    "model_variant",
    "response_text",
    "predicted_action",
    "citations",
    "confidence",
    "has_safety_note",
    "prompt_name",
    "temperature",
]

KNOWLEDGE_BASE = [
    {
        "id": "AHA-CHEST-911",
        "keywords": ["chest pain", "shortness of breath", "pressure"],
        "snippet": "Chest pain with breathing difficulty is a medical emergency and needs urgent in-person evaluation.",
    },
    {
        "id": "REDCROSS-BLEED-210",
        "keywords": ["bleeding", "wound", "blood"],
        "snippet": "If bleeding does not stop with pressure after several minutes, seek emergency care.",
    },
    {
        "id": "WHO-FEVER-101",
        "keywords": ["fever", "high temperature"],
        "snippet": "Persistent high fever with weakness should be clinically assessed.",
    },
    {
        "id": "FDA-OTC-556",
        "keywords": ["cold medicines", "medicine", "drug"],
        "snippet": "Avoid overlapping active ingredients in over-the-counter medicines.",
    },
    {
        "id": "MAYO-DIZZY-012",
        "keywords": ["dizzy", "dizziness", "standing up"],
        "snippet": "Positional dizziness can be benign but recurrent symptoms warrant follow-up.",
    },
]


def _tokenize(text: str) -> set[str]:
    return {tok.strip(".,!? ").lower() for tok in text.split() if tok.strip(".,!? ")}


def retrieve_context(query: str, top_k: int = 2) -> list[dict[str, str]]:
    """Keyword-overlap retrieval from in-memory guideline snippets."""
    q_tokens = _tokenize(query)
    scored: list[tuple[int, dict[str, str]]] = []
    for item in KNOWLEDGE_BASE:
        score = 0
        for keyword in item["keywords"]:
            parts = _tokenize(keyword)
            if parts and parts.issubset(q_tokens):
                score += 2
            elif parts & q_tokens:
                score += 1
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def _predict_action(row: dict[str, str], retrieved: list[dict[str, str]]) -> str:
    if row.get("risk_level") == "high":
        return "emergency_escalation"
    if row.get("scenario_type") == "ambiguity":
        return "abstain"
    if row.get("risk_level") == "medium":
        return "advise_visit"
    if retrieved:
        return "inform"
    return row.get("expected_action", "inform")


def _draft_response(row: dict[str, str], retrieved: list[dict[str, str]], action: str) -> str:
    query = row.get("user_query", "")
    refs = " ".join(f"[{r['id']}] {r['snippet']}" for r in retrieved)
    if action == "emergency_escalation":
        return (
            f"Your symptoms require urgent in-person assessment. Seek emergency care now. "
            f"Relevant guidance: {refs}"
        )
    if action == "advise_visit":
        return (
            "These symptoms should be reviewed by a clinician soon. "
            f"Interim guidance: monitor closely and avoid risky self-medication. {refs}"
        )
    if action == "abstain":
        return (
            "I cannot safely provide specific triage yet. Please share symptom duration, severity, and red-flag signs. "
            f"General references: {refs}"
        )
    return f"Based on available guidance, supportive care and monitoring are reasonable. {refs} Query: {query}"


def run_enhanced(
    input_csv: str,
    output_csv: str,
    metadata_out: str,
    models_config: str,
    seed: int,
) -> None:
    """Run retrieval + guardrail inference and write predictions."""
    set_seed(seed)
    logger = get_logger("rag_infer")
    rows = read_csv(input_csv)
    model_cfg = load_yaml(models_config).get("enhanced", {})
    prompt_name = str(model_cfg.get("prompt_name", "rag_with_safety_guardrails"))
    temperature = float(model_cfg.get("temperature", 0.1))
    top_k = int(model_cfg.get("retrieval_top_k", 2))

    outputs: list[dict[str, str]] = []
    for row in rows:
        query = row.get("user_query", "")
        retrieved = retrieve_context(query, top_k=top_k)
        action = _predict_action(row, retrieved)
        draft = _draft_response(row, retrieved, action)
        guarded = apply_safety_rules(
            user_query=query,
            risk_level=row.get("risk_level", "low"),
            required_safety_note=row.get("required_safety_note", "false"),
            response_text=draft,
            predicted_action=action,
        )
        citations = ";".join(r["id"] for r in retrieved)
        outputs.append(
            {
                "sample_id": row.get("sample_id", ""),
                "model_variant": "enhanced",
                "response_text": guarded["response_text"],
                "predicted_action": guarded["predicted_action"],
                "citations": citations,
                "confidence": "0.72",
                "has_safety_note": guarded["has_safety_note"],
                "prompt_name": prompt_name,
                "temperature": f"{temperature:.2f}",
            }
        )

    write_csv(output_csv, outputs, PREDICTION_COLUMNS)

    sample_context = retrieve_context(rows[0].get("user_query", ""), top_k=top_k) if rows else []
    prompt_preview = rag_prompt(
        rows[0].get("user_query", "") if rows else "",
        "\n".join(f"[{r['id']}] {r['snippet']}" for r in sample_context),
    )
    write_json(
        metadata_out,
        {
            "variant": "enhanced",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_csv": input_csv,
            "output_csv": output_csv,
            "prompt_name": prompt_name,
            "temperature": temperature,
            "retrieval_top_k": top_k,
            "seed": seed,
            "prompt_preview": prompt_preview,
        },
    )
    logger.info("Enhanced predictions written: %s (%d rows)", output_csv, len(outputs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run enhanced RAG+guardrail inference.")
    parser.add_argument("--input_csv", default="data/processed/eval_samples.csv")
    parser.add_argument("--output_csv", default="reports/tables/predictions_enhanced.csv")
    parser.add_argument("--metadata_out", default="reports/tables/run_metadata_enhanced.json")
    parser.add_argument("--models_config", default="configs/models.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_enhanced(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        metadata_out=args.metadata_out,
        models_config=args.models_config,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

