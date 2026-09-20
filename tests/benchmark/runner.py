"""
Shared logic between test_ragas_benchmark.py (the CI regression check)
and update_baseline.py (the manual "accept new baseline" script) — both
need to do the exact same thing: ingest the fixed QA set, run the RAG
agent against each question, and average the RAGAS scores across items.

Requires a live Weaviate instance and a working LLM provider API key —
this is NOT part of the default unit test run.
"""
import json
from pathlib import Path
from statistics import mean

from rag_agent_service.app.agent import run_rag_agent
from rag_agent_service.app.ingestion import ingest_texts

BENCHMARK_DIR = Path(__file__).parent
QA_SET_PATH = BENCHMARK_DIR / "qa_set.json"
BASELINE_PATH = BENCHMARK_DIR / "ragas_baseline.json"


def load_qa_set() -> list[dict]:
    return json.loads(QA_SET_PATH.read_text())


def load_baseline() -> dict:
    data = json.loads(BASELINE_PATH.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def ingest_qa_set_documents(qa_set: list[dict]) -> None:
    ingest_texts([(item["document_text"], item["source"]) for item in qa_set])


def run_benchmark() -> dict[str, float]:
    """Ingests the fixed corpus, runs the RAG agent on every question,
    and returns the mean of each RAGAS metric across all questions."""
    qa_set = load_qa_set()
    ingest_qa_set_documents(qa_set)

    per_metric_scores: dict[str, list[float]] = {}
    for item in qa_set:
        result = run_rag_agent(item["question"], trace_id="benchmark")
        for metric, score in (result.get("ragas_scores") or {}).items():
            if isinstance(score, (int, float)):
                per_metric_scores.setdefault(metric, []).append(score)

    return {metric: round(mean(scores), 4) for metric, scores in per_metric_scores.items() if scores}
