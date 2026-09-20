"""
Tool agent core logic — called by server.py. The LLM proposes tool calls
via function-calling; nothing it proposes runs directly — every call goes
through validate_and_dispatch() first (schema + domain allow-list checks).
"""
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from shared.core.logging import log_step
from shared.core.prompts import load_prompt
from shared.services.chat_llm import get_chat_llm, invoke_and_track
from tool_agent_service.app.guardrails import guard_input, guard_output
from tool_agent_service.app.validation import validate_and_dispatch

PROMPT_PATH = Path(__file__).parent / "prompts" / "tool_system_prompt.md"
MAX_TOOL_ITERATIONS = 4


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Use for questions about recent events,
    facts you're unsure of, or anything requiring up-to-date information."""
    return ""  # schema only — see run_tool_agent, execution goes through validate_and_dispatch


@tool
def fetch_url(url: str) -> str:
    """Fetch and extract the readable text content of a specific URL."""
    return ""  # schema only


def run_tool_agent(query: str, trace_id: str, model_override: dict | None = None) -> dict:
    """
    model_override: {"provider": ..., "model": ...} — lets a caller (the
    Streamlit UI's model selector, via the orchestrator) switch LLM
    provider/model per-request instead of using this service's .env
    default. Note: the selected model must support function-calling
    (bind_tools) for the tool-call loop below to work — this isn't
    validated here, it'll simply fail at the LLM call if unsupported.

    Returns:
      answer: str
      tool_calls_made: list[dict]
      guardrail_flags: list[str]
      blocked: bool
      block_reason: str | None
    """
    input_guard = guard_input(query)
    if not input_guard.allowed:
        return {
            "answer": "", "tool_calls_made": [],
            "guardrail_flags": input_guard.flags, "blocked": True, "block_reason": input_guard.reason,
        }

    override = model_override or {}
    system_prompt = load_prompt(str(PROMPT_PATH))
    llm = get_chat_llm(
        temperature=0.2,
        provider_override=override.get("provider"),
        model_override=override.get("model"),
    ).bind_tools([web_search, fetch_url])
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
    tool_calls_made = []

    with log_step("run_tool_agent", agent="tool_agent", query=query) as ctx:
        final_text = ""
        for _ in range(MAX_TOOL_ITERATIONS):
            response: AIMessage = invoke_and_track(llm, messages, node="tool_agent_step", agent="tool_agent")
            messages.append(response)

            if not response.tool_calls:
                final_text = response.content
                break

            for call in response.tool_calls:
                validated = validate_and_dispatch(call["name"], call["args"])
                tool_calls_made.append({
                    "name": call["name"], "args": call["args"],
                    "ok": validated.ok, "error": validated.error,
                })
                content = validated.result if validated.ok else f"ERROR: {validated.error}"
                messages.append(ToolMessage(content=str(content), tool_call_id=call["id"]))
        else:
            final_response = invoke_and_track(
                llm, messages + [HumanMessage(content="Provide your best final answer now based on the information gathered.")],
                node="tool_agent_forced_final", agent="tool_agent",
            )
            final_text = final_response.content

        ctx["output"] = {"tool_calls_made": len(tool_calls_made)}

    output_guard = guard_output(final_text if isinstance(final_text, str) else str(final_text))
    final_answer = output_guard.redacted_text if output_guard.allowed else ""

    return {
        "answer": final_answer,
        "tool_calls_made": tool_calls_made,
        "guardrail_flags": output_guard.flags,
        "blocked": not output_guard.allowed,
        "block_reason": output_guard.reason,
    }
