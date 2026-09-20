from shared.core.guardrails import GuardrailResult, check_input, check_output


def guard_input(query: str) -> GuardrailResult:
    return check_input(query)


def guard_output(answer: str) -> GuardrailResult:
    # Tool-fetched content is the main injection surface here (a fetched
    # webpage could itself contain "ignore previous instructions"-style
    # text) — the LLM's final synthesized answer is what we guard, not
    # the raw fetched page content, since raw pages are never returned
    # to the user directly (see tools.py / agent.py).
    return check_output(answer)
