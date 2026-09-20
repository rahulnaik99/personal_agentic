"""
Explicit validation for tool calling.

The LLM's raw tool-call output (name + args) is never executed directly.
Every call goes through `validate_and_dispatch()`, which:
  1. Confirms the tool name is in the registry (no arbitrary code paths)
  2. Validates args against a strict Pydantic schema per tool
  3. Applies domain allow-listing / URL sanity checks for fetch_url
  4. Only then dispatches to the actual tool function

Any validation failure returns a structured error instead of raising —
the orchestrator/agent can decide how to react (retry, ask user, skip).
"""
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, ValidationError, field_validator

from shared.core.config import get_settings
from shared.core.logging import log_step
from tool_agent_service.app.tools import TOOL_REGISTRY


class WebSearchArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


class FetchUrlArgs(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def must_be_http_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"Invalid URL: {v}")
        return v


TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "web_search": WebSearchArgs,
    "fetch_url": FetchUrlArgs,
}


class ToolCallRequest(BaseModel):
    tool_name: Literal["web_search", "fetch_url"]
    args: dict[str, Any]


class ToolCallResult(BaseModel):
    ok: bool
    tool_name: str
    result: str | list | dict | None = None
    error: str | None = None


def _is_domain_allowed(url: str) -> bool:
    settings = get_settings()
    if not settings.TOOL_ALLOWED_DOMAINS:
        return True  # no allow-list configured -> unrestricted
    allowed = {d.strip().lower() for d in settings.TOOL_ALLOWED_DOMAINS.split(",") if d.strip()}
    host = urlparse(url).netloc.lower()
    return any(host == d or host.endswith(f".{d}") for d in allowed)


def validate_and_dispatch(tool_name: str, raw_args: dict[str, Any]) -> ToolCallResult:
    with log_step("validate_and_dispatch", agent="tool_agent", tool_name=tool_name, raw_args=raw_args) as ctx:
        # 1. Tool exists in registry
        if tool_name not in TOOL_REGISTRY or tool_name not in TOOL_SCHEMAS:
            result = ToolCallResult(ok=False, tool_name=tool_name, error=f"Unknown tool: {tool_name}")
            ctx["output"] = result.model_dump()
            return result

        # 2. Schema validation
        schema_cls = TOOL_SCHEMAS[tool_name]
        try:
            validated_args = schema_cls(**raw_args)
        except ValidationError as exc:
            result = ToolCallResult(ok=False, tool_name=tool_name, error=f"Invalid args: {exc}")
            ctx["output"] = result.model_dump()
            return result

        # 3. Domain allow-list check (fetch_url only)
        if tool_name == "fetch_url":
            url = validated_args.url  # type: ignore[attr-defined]
            if not _is_domain_allowed(url):
                result = ToolCallResult(ok=False, tool_name=tool_name, error=f"Domain not allowed: {url}")
                ctx["output"] = result.model_dump()
                return result

        # 4. Dispatch
        try:
            fn = TOOL_REGISTRY[tool_name]
            output = fn(**validated_args.model_dump())
            result = ToolCallResult(ok=True, tool_name=tool_name, result=output)
        except Exception as exc:
            result = ToolCallResult(ok=False, tool_name=tool_name, error=str(exc))

        ctx["output"] = {"ok": result.ok, "error": result.error}
        return result
