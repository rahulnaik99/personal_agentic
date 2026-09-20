
from shared.core.config import get_settings
from shared.core.guardrails import GuardrailResult, check_input, check_output


def guard_input(query: str) -> GuardrailResult:
    return check_input(query)


def guard_output(answer: str, ragas_scores: dict | None = None) -> GuardrailResult:
    result = check_output(answer)
    if not result.allowed:
        return result

    settings = get_settings()
    faithfulness = (ragas_scores or {}).get("faithfulness")
    if isinstance(faithfulness, (int, float)) and faithfulness < settings.GUARDRAILS_LOW_FAITHFULNESS_THRESHOLD:
        result.flags.append("low_faithfulness")
        # Not blocked — flagged so the orchestrator/UI can surface a
        # "low confidence" notice rather than silently returning a
        # possibly-ungrounded answer.
    return result
