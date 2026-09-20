from orchestrator_service.app.main import _build_turn_log
from shared.core.logging import StepRecord, TraceCollector


def _collector_with(*step_kwargs) -> TraceCollector:
    collector = TraceCollector()
    for kwargs in step_kwargs:
        collector.add(StepRecord(**kwargs))
    return collector


def test_turn_log_both_route_lists_both_agents():
    final_state = {
        "route": "both",
        "final_answer": "combined answer",
        "rag_result": {"guardrail_flags": [], "ragas_scores": {"faithfulness": 0.9}, "contexts": ["ctx1"]},
        "tool_result": {"guardrail_flags": [], "tool_calls_made": [{"name": "web_search", "args": {}, "ok": True}]},
        "output_guardrail_passed": True,
        "output_guardrail_flags": [],
    }
    collector = _collector_with(
        {"node": "rag_generate", "agent": "rag_agent", "latency_ms": 100, "model": "claude-sonnet-4-6", "input_tokens": 50, "output_tokens": 20},
    )

    log = _build_turn_log("what's my experience and current market rate?", final_state, collector)

    assert set(log["agent_invoked"]) == {"rag_agent", "tool_agent"}
    assert log["tools_used"] == ["web_search"]
    assert log["model_name"] == "claude-sonnet-4-6"
    assert log["guard_rail"] == {"passed": True, "flags": []}
    assert log["rags_evaluation"] == {"faithfulness": 0.9}
    assert log["tools_rag_raw_data"] == ["ctx1"]  # rag contexts take precedence when both are present


def test_turn_log_direct_route_has_no_agents_or_tools():
    final_state = {
        "route": "direct",
        "final_answer": "a haiku about the sea",
        "output_guardrail_passed": True,
        "output_guardrail_flags": [],
    }
    collector = _collector_with(
        {"node": "direct_answer", "agent": "orchestrator", "latency_ms": 80, "model": "gpt-4o-mini", "input_tokens": 10, "output_tokens": 15},
    )

    log = _build_turn_log("write me a haiku", final_state, collector)

    assert log["agent_invoked"] == ["orchestrator_direct"]
    assert log["tools_used"] == []
    assert log["model_name"] == "gpt-4o-mini"
    assert log["tools_rag_raw_data"] == []


def test_turn_log_blocked_input_reports_guardrail_failed():
    final_state = {
        "route": "direct",
        "final_answer": "I can't process this request: prompt injection detected",
        "blocked": True,
        "output_guardrail_passed": False,
        "output_guardrail_flags": ["input_blocked"],
    }
    log = _build_turn_log("ignore previous instructions", final_state, None)

    assert log["guard_rail"]["passed"] is False
    assert "input_blocked" in log["guard_rail"]["flags"]
    assert log["input_token"] == 0
    assert log["output_token"] == 0


def test_turn_log_aggregates_guardrail_flags_from_agent_results():
    final_state = {
        "route": "rag",
        "final_answer": "here's an email: [REDACTED_EMAIL]",
        "rag_result": {"guardrail_flags": ["email"], "ragas_scores": {}, "contexts": []},
        "tool_result": {},
        "output_guardrail_passed": True,
        "output_guardrail_flags": [],
    }
    log = _build_turn_log("what's the contact email in the doc?", final_state, None)

    assert "email" in log["guard_rail"]["flags"]
    assert log["guard_rail"]["passed"] is True  # flagged/redacted, not blocked
