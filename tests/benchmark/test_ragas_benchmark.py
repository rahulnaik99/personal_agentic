"""
Regression test: the RAG agent's RAGAS scores on the fixed benchmark
set must not drop more than RAGAS_REGRESSION_TOLERANCE below the stored
baseline. Requires live Weaviate + a working LLM API key — this file is
NOT part of the default `pytest tests/unit` run; it's invoked as its own
CI job (see .github/workflows/ci.yml), gated on required secrets being
present.

Updating the baseline itself is a deliberate, separate, manual step —
see update_baseline.py — so a real regression can never silently reset
the bar it's being checked against.
"""
import pytest

from shared.core.config import get_settings
from tests.benchmark.runner import load_baseline, run_benchmark


@pytest.fixture(scope="module")
def benchmark_scores():
    return run_benchmark()


def test_ragas_scores_do_not_regress(benchmark_scores):
    settings = get_settings()
    baseline = load_baseline()
    tolerance = settings.RAGAS_REGRESSION_TOLERANCE

    failures = []
    for metric, baseline_score in baseline.items():
        current_score = benchmark_scores.get(metric)
        if current_score is None:
            failures.append(f"{metric}: no current score produced (baseline={baseline_score})")
            continue
        if current_score < baseline_score - tolerance:
            failures.append(
                f"{metric}: {current_score} is more than {tolerance} below baseline {baseline_score}"
            )

    assert not failures, "RAGAS regression detected:\n" + "\n".join(failures)
