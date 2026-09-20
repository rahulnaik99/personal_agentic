"""
Common element type produced by every loader (PDF, TXT, future formats),
consumed by chunking.py. Keeping loaders and chunking decoupled through
this one small type is what lets chunking treat "a table from a PDF" and
"a table pasted into a plain text doc" the same way.
"""
from dataclasses import dataclass
from typing import Literal

ElementType = Literal["text", "table", "image"]


@dataclass
class DocElement:
    type: ElementType
    text: str                       # for "table": markdown/HTML table text; for "image": caption
    image_path: str | None = None  # set only for type == "image"
