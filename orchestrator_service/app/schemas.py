from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ModelOverride(BaseModel):
    provider: Literal["local", "mlx", "openai", "anthropic"]
    model: str | None = None


class RouteDecision(BaseModel):
    route: Literal["rag", "tool", "both", "direct"]
    reason: str = Field(default="")


class GraphState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    query: str
    trace_id: str
    model_override: dict[str, Any] | None
    conversation_history: list[dict[str, str]]

    route: str
    routing_reason: str
    loop_count: int

    rag_result: dict[str, Any] | None
    tool_result: dict[str, Any] | None

    final_answer: str | None
    blocked: bool
    block_reason: str | None
    output_guardrail_passed: bool
    output_guardrail_flags: list[str]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: str | None = None
    model_override: ModelOverride | None = None
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    route: str
    trace_id: str
    ragas_scores: dict | None = None
    tool_calls_made: list | None = None
    trace: list | None = None
    total_cost_usd: float | None = None
    total_tokens: dict | None = None
    turn_log: dict | None = None
