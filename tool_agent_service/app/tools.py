"""
Free tools for the tool agent:
  - web_search: DuckDuckGo search, no API key required
  - fetch_url: fetches a URL and extracts readable text content

These are plain callables (not decorated as LangChain @tool here) because
they're invoked through the explicit validation layer in validation.py
first — the orchestration layer controls exactly when/how they run
rather than letting the LLM call them unchecked.
"""
import httpx
from ddgs import DDGS

from shared.core.config import get_settings
from shared.core.logging import log_step


def web_search(query: str) -> list[dict]:
    settings = get_settings()
    with log_step("web_search", agent="tool_agent", query=query) as ctx:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=settings.WEB_SEARCH_MAX_RESULTS))
        formatted = [
            {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
            for r in results
        ]
        ctx["output"] = {"num_results": len(formatted)}
    return formatted


def fetch_url(url: str) -> str:
    settings = get_settings()
    with log_step("fetch_url", agent="tool_agent", url=url) as ctx:
        try:
            with httpx.Client(timeout=settings.URL_FETCH_TIMEOUT_SECONDS, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "Mozilla/5.0 (agentic-app)"})
                resp.raise_for_status()
                html = resp.text
        except Exception as exc:
            ctx["output"] = {"error": str(exc)}
            return f"ERROR fetching {url}: {exc}"

        text = _extract_readable_text(html)
        truncated = text[: settings.URL_FETCH_MAX_CHARS]
        ctx["output"] = {"chars_extracted": len(truncated)}
        return truncated


def _extract_readable_text(html: str) -> str:
    try:
        import trafilatura
        extracted = trafilatura.extract(html)
        if extracted:
            return extracted
    except Exception:
        pass
    # Fallback: crude tag stripping if trafilatura fails/unavailable
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


TOOL_REGISTRY = {
    "web_search": web_search,
    "fetch_url": fetch_url,
}
