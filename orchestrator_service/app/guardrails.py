from shared.core.guardrails import GuardrailResult, check_input, check_output


def guard_input(query: str) -> GuardrailResult:
    return check_input(query)


def guard_output(answer: str) -> GuardrailResult:
    return check_output(answer)
