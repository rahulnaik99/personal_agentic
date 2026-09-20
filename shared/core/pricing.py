"""
Rough $/1M-token pricing per model, used only to show an estimated cost
in the Streamlit "thinking" trace — NOT wired into any billing system.
Update PRICING_TABLE as providers change prices; unknown models fall
back to a configurable default so the UI never crashes on a new model
name, it just shows an approximate/blank cost.
"""

# (input $/1M tokens, output $/1M tokens)
PRICING_TABLE: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Local models are effectively free (compute cost not tracked here)
    "llama3.1": (0.0, 0.0),
}

DEFAULT_PRICING = (0.0, 0.0)  # unknown model -> cost shown as 0 rather than guessed


def estimate_cost_usd(model: str | None, input_tokens: int, output_tokens: int) -> float:
    if not model:
        return 0.0
    input_price, output_price = PRICING_TABLE.get(model, DEFAULT_PRICING)
    cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
    return round(cost, 6)
