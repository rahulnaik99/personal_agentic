"""
Parent-child chunking, element-aware.

Text elements accumulate into a buffer and get split the same way as
before (large "parent" context blocks, further split into small "child"
search chunks). Table and image elements are never split — a table
mid-row or an image caption chopped in half loses meaning — each becomes
its own single parent+child pair, tagged with element_type so retrieval/
generation can treat it specially (e.g. render a table's HTML verbatim
in the LLM context rather than paraphrasing it).

Every chunk also carries `category` (user-supplied at ingestion time,
e.g. "profession_doc") so retrieval can filter to the right document set
— see rag_agent_service/app/agent.py's classify_category().
"""
import uuid
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent_service.app.loaders.base import DocElement
from shared.core.config import get_settings

UNCATEGORIZED = "uncategorized"


@dataclass
class ParentChunk:
    id: str
    text: str
    source: str
    element_type: str = "text"
    category: str = UNCATEGORIZED


@dataclass
class ChildChunk:
    id: str
    text: str
    parent_id: str
    source: str
    chunk_index: int
    element_type: str = "text"
    image_path: str | None = None
    category: str = UNCATEGORIZED


@dataclass
class ChunkedDocument:
    parents: list[ParentChunk] = field(default_factory=list)
    children: list[ChildChunk] = field(default_factory=list)


def _flush_text_buffer(buffer: str, source: str, category: str, result: ChunkedDocument) -> None:
    if not buffer.strip():
        return
    settings = get_settings()
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.PARENT_CHUNK_SIZE, chunk_overlap=settings.PARENT_CHUNK_OVERLAP,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHILD_CHUNK_SIZE, chunk_overlap=settings.CHILD_CHUNK_OVERLAP,
    )

    for parent_text in parent_splitter.split_text(buffer):
        parent_id = str(uuid.uuid4())
        result.parents.append(ParentChunk(
            id=parent_id, text=parent_text, source=source, element_type="text", category=category,
        ))
        for idx, child_text in enumerate(child_splitter.split_text(parent_text)):
            result.children.append(ChildChunk(
                id=str(uuid.uuid4()), text=child_text, parent_id=parent_id,
                source=source, chunk_index=idx, element_type="text", category=category,
            ))


def _add_whole_element(element: DocElement, source: str, category: str, result: ChunkedDocument) -> None:
    """Tables and images: parent == child == the whole element, unsplit."""
    parent_id = str(uuid.uuid4())
    result.parents.append(ParentChunk(
        id=parent_id, text=element.text, source=source, element_type=element.type, category=category,
    ))
    result.children.append(ChildChunk(
        id=str(uuid.uuid4()), text=element.text, parent_id=parent_id,
        source=source, chunk_index=0, element_type=element.type,
        image_path=element.image_path, category=category,
    ))


def chunk_elements(elements: list[DocElement], source: str, category: str = UNCATEGORIZED) -> ChunkedDocument:
    result = ChunkedDocument()
    text_buffer = ""

    for element in elements:
        if element.type == "text":
            text_buffer += ("\n\n" if text_buffer else "") + element.text
            continue

        # Non-text element: flush accumulated text first, then add it whole.
        _flush_text_buffer(text_buffer, source, category, result)
        text_buffer = ""
        _add_whole_element(element, source, category, result)

    _flush_text_buffer(text_buffer, source, category, result)
    return result


def chunk_document(text: str, source: str, category: str = UNCATEGORIZED) -> ChunkedDocument:
    """Backwards-compatible entrypoint for plain raw text (no file/elements)."""
    return chunk_elements([DocElement(type="text", text=text)], source, category)
