"""
Plain .txt loader — no partitioning needed, the whole file is one text
element and flows through the normal parent-child splitter in chunking.py.
"""

from rag_agent_service.app.loaders.base import DocElement


def load_txt(file_path: str, source: str) -> list[DocElement]:
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    return [DocElement(type="text", text=text)] if text else []
