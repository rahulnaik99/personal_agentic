from shared.core.pricing import estimate_cost_usd


def test_known_model_computes_expected_cost():
    # claude-sonnet-4-6: $3/1M input, $15/1M output
    cost = estimate_cost_usd("claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 18.0


def test_unknown_model_returns_zero():
    assert estimate_cost_usd("some-brand-new-model", input_tokens=1000, output_tokens=1000) == 0.0


def test_none_model_returns_zero():
    assert estimate_cost_usd(None, input_tokens=1000, output_tokens=1000) == 0.0


def test_zero_tokens_returns_zero_cost():
    assert estimate_cost_usd("claude-sonnet-4-6", input_tokens=0, output_tokens=0) == 0.0


def test_local_model_is_free():
    assert estimate_cost_usd("llama3.1", input_tokens=50_000, output_tokens=50_000) == 0.0
