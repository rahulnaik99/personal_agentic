"""
RAGAS evaluation hook.

Runs after every RAG generation (if RAGAS_ENABLED) and logs the scores
via the standard structured logger — this is diagnostic instrumentation,
not a gate that blocks the response, so a low score never fails the
request; it just gets logged for later analysis.
"""

from shared.core.config import get_settings
from shared.core.logging import log_step


def evaluate_rag_response(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str | None = None,
) -> dict:
    settings = get_settings()
    if not settings.RAGAS_ENABLED:
        return {}

    with log_step("ragas_evaluate", agent="rag_agent", question=question) as ctx:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            metrics = [faithfulness, answer_relevancy, context_precision]
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]
                metrics.append(context_recall)

            dataset = Dataset.from_dict(data)
            result = evaluate(dataset, metrics=metrics)
            scores = result.to_pandas().iloc[0].to_dict()
            ctx["output"] = scores
            return scores
        except Exception as exc:  # RAGAS/eval failures must never break the chat response
            ctx["output"] = {"error": str(exc)}
            return {"error": str(exc)}
