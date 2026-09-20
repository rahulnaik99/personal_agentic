"""
Loads a prompt from a .md file. Optional YAML frontmatter can carry
metadata (e.g. `output_schema: json`); the body (everything after the
frontmatter) is the actual prompt text handed to the LLM.

Usage:
    prompt = load_prompt(Path(__file__).parent / "prompts" / "rag_system_prompt.md")
"""
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=64)
def load_prompt(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if text.startswith("---"):
        # Strip a YAML frontmatter block if present; we only need the body.
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text.strip()
