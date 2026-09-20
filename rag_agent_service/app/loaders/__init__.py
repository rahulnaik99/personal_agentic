import os

from rag_agent_service.app.loaders.base import DocElement
from rag_agent_service.app.loaders.pdf_loader import load_pdf
from rag_agent_service.app.loaders.txt_loader import load_txt

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_file(file_path: str, source: str) -> list[DocElement]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path, source)
    if ext == ".txt":
        return load_txt(file_path, source)
    raise ValueError(f"Unsupported file extension: {ext} (supported: {sorted(SUPPORTED_EXTENSIONS)})")
