import pytest

from shared.core.config import get_settings
from tool_agent_service.app.validation import validate_and_dispatch


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_unknown_tool_name_rejected():
    result = validate_and_dispatch("not_a_real_tool", {})
    assert not result.ok
    assert "Unknown tool" in result.error


def test_web_search_invalid_args_rejected():
    result = validate_and_dispatch("web_search", {"query": ""})
    assert not result.ok
    assert "Invalid args" in result.error


def test_web_search_valid_args_dispatches(monkeypatch):
    monkeypatch.setattr(
        "tool_agent_service.app.validation.TOOL_REGISTRY",
        {"web_search": lambda query: [{"title": "fake", "url": "https://example.com", "snippet": "..."}],
         "fetch_url": lambda url: "fake content"},
    )
    result = validate_and_dispatch("web_search", {"query": "test query"})
    assert result.ok
    assert result.result[0]["title"] == "fake"


def test_fetch_url_rejects_non_http_scheme():
    result = validate_and_dispatch("fetch_url", {"url": "ftp://example.com/file"})
    assert not result.ok
    assert "Invalid args" in result.error


def test_fetch_url_respects_domain_allowlist(monkeypatch):
    monkeypatch.setenv("TOOL_ALLOWED_DOMAINS", "trusted.com")
    get_settings.cache_clear()

    monkeypatch.setattr(
        "tool_agent_service.app.validation.TOOL_REGISTRY",
        {"fetch_url": lambda url: "should not be called"},
    )

    blocked = validate_and_dispatch("fetch_url", {"url": "https://untrusted.com/page"})
    assert not blocked.ok
    assert "Domain not allowed" in blocked.error

    allowed = validate_and_dispatch("fetch_url", {"url": "https://trusted.com/page"})
    assert allowed.ok


def test_fetch_url_subdomain_matches_allowlist(monkeypatch):
    monkeypatch.setenv("TOOL_ALLOWED_DOMAINS", "trusted.com")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "tool_agent_service.app.validation.TOOL_REGISTRY",
        {"fetch_url": lambda url: "ok"},
    )
    result = validate_and_dispatch("fetch_url", {"url": "https://docs.trusted.com/page"})
    assert result.ok


def test_tool_execution_exception_is_captured_not_raised(monkeypatch):
    def boom(query):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(
        "tool_agent_service.app.validation.TOOL_REGISTRY",
        {"web_search": boom, "fetch_url": lambda url: "ok"},
    )
    result = validate_and_dispatch("web_search", {"query": "anything"})
    assert not result.ok
    assert "network exploded" in result.error
