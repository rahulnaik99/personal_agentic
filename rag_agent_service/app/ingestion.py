"""
Ingestion pipeline. Two entrypoints:
  - ingest_files(): PDF/TXT paths -> loader -> chunk_elements -> Weaviate
  - ingest_texts(): raw (text, source) pairs -> chunk_document -> Weaviate
    (kept for backwards compatibility / quick testing without a file)

Both now handle re-ingestion of the same `source` deliberately rather
than blindly duplicating:
  - unchanged: content_hash matches what's already active -> skip entirely
  - updated:   content_hash differs -> deactivate the old chunks (never
               deleted, so history stays recoverable) and insert the new
               ones as active
  - first_ingest: no active chunks exist yet for this `source`

Every chunk also carries a user-supplied `category` (e.g. "profession_doc",
"financial_doc") so retrieval can filter to the right document set —
see rag_agent_service/app/agent.py's classify_category().
"""
import hashlib

from rag_agent_service.app.chunking import ChunkedDocument, chunk_document, chunk_elements
from rag_agent_service.app.loaders import load_file
from shared.core.config import get_settings
from shared.core.logging import log_step
from shared.services.encoder_llm import get_encoder
from shared.vectorstore.weaviate_client import (
    deactivate_source,
    ensure_schema,
    get_existing_chunk_hash,
    get_weaviate_client,
    now_iso,
)


def _hash_content(content: str | bytes) -> str:
    data = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(data).hexdigest()


def _resolve_action(source: str, content_hash: str) -> str:
    """Returns "first_ingest" | "unchanged" | "updated" for this source."""
    existing_hash = get_existing_chunk_hash(source)
    if existing_hash is None:
        return "first_ingest"
    if existing_hash == content_hash:
        return "unchanged"
    return "updated"


def _upsert_chunked_document(chunked: ChunkedDocument, content_hash: str, encoder, parent_collection, child_collection) -> tuple[int, int]:
    ingested_at = now_iso()

    with parent_collection.batch.dynamic() as batch:
        for parent in chunked.parents:
            batch.add_object(
                properties={
                    "text": parent.text, "source": parent.source, "metadata_json": "{}",
                    "category": parent.category, "is_active": True,
                    "ingested_at": ingested_at, "content_hash": content_hash,
                },
                uuid=parent.id,
            )

    if not chunked.children:
        return len(chunked.parents), 0

    # Only text/table children need a dense vector computed from their own
    # content; image children get a vector from their caption text too —
    # `child.text` already holds the caption, so this is uniform either way.
    child_vectors = encoder.embed_documents([c.text for c in chunked.children])
    with child_collection.batch.dynamic() as batch:
        for child, vector in zip(chunked.children, child_vectors, strict=True):
            batch.add_object(
                properties={
                    "text": child.text,
                    "parent_id": child.parent_id,
                    "source": child.source,
                    "chunk_index": child.chunk_index,
                    "element_type": child.element_type,
                    "image_path": child.image_path or "",
                    "category": child.category,
                    "is_active": True,
                    "ingested_at": ingested_at,
                    "content_hash": content_hash,
                },
                uuid=child.id,
                vector=vector,
            )
    return len(chunked.parents), len(chunked.children)


def ingest_file_with_progress(file_path: str, source: str, category: str):
    """
    Generator variant of a single-file ingest, yielding a progress dict at
    each real stage boundary — used by ingest_api.py's SSE endpoint
    (POST /ingest/file/stream) to drive the UI's live checklist:
    loading_file -> checking_duplicate -> chunking -> embedding_and_storing
    -> completed.

    Order matters for efficiency: the file is only actually partitioned
    (load_file() — expensive for PDFs) if it turns out NOT to be an
    unchanged duplicate, so a re-upload of the same content short-circuits
    right after the hash check instead of wastefully re-partitioning it.
    """
    settings = get_settings()
    ensure_schema()
    client = get_weaviate_client()
    encoder = get_encoder()
    parent_collection = client.collections.get(settings.WEAVIATE_CLASS_PARENT)
    child_collection = client.collections.get(settings.WEAVIATE_CLASS_CHILD)

    yield {"stage": "loading_file", "status": "start"}
    with open(file_path, "rb") as f:
        content_hash = _hash_content(f.read())
    yield {"stage": "loading_file", "status": "done"}

    yield {"stage": "checking_duplicate", "status": "start"}
    action = _resolve_action(source, content_hash)
    yield {"stage": "checking_duplicate", "status": "done", "action": action}

    if action == "unchanged":
        yield {
            "stage": "completed", "status": "done",
            "result": {"source": source, "status": "unchanged", "parents": 0, "children": 0},
        }
        return
    if action == "updated":
        deactivate_source(source)

    yield {"stage": "chunking", "status": "start"}
    elements = load_file(file_path, source)
    chunked = chunk_elements(elements, source, category)
    yield {"stage": "chunking", "status": "done", "num_parents": len(chunked.parents), "num_children": len(chunked.children)}

    yield {"stage": "embedding_and_storing", "status": "start"}
    parents, children = _upsert_chunked_document(chunked, content_hash, encoder, parent_collection, child_collection)
    yield {"stage": "embedding_and_storing", "status": "done"}

    yield {
        "stage": "completed", "status": "done",
        "result": {"source": source, "status": action, "parents": parents, "children": children},
    }


def ingest_files(file_paths_sources_categories: list[tuple[str, str, str]]) -> dict:
    """file_paths_sources_categories: list of (local_file_path, source_label, category) tuples."""
    settings = get_settings()
    ensure_schema()
    client = get_weaviate_client()
    encoder = get_encoder()
    parent_collection = client.collections.get(settings.WEAVIATE_CLASS_PARENT)
    child_collection = client.collections.get(settings.WEAVIATE_CLASS_CHILD)

    total_parents = total_children = 0
    results = []
    with log_step("ingest_files", agent="rag_agent", num_files=len(file_paths_sources_categories)) as ctx:
        for file_path, source, category in file_paths_sources_categories:
            with open(file_path, "rb") as f:
                content_hash = _hash_content(f.read())

            action = _resolve_action(source, content_hash)
            if action == "unchanged":
                results.append({"source": source, "status": "unchanged", "parents": 0, "children": 0})
                continue
            if action == "updated":
                deactivate_source(source)

            elements = load_file(file_path, source)
            chunked = chunk_elements(elements, source, category)
            p, c = _upsert_chunked_document(chunked, content_hash, encoder, parent_collection, child_collection)
            total_parents += p
            total_children += c
            results.append({"source": source, "status": action, "parents": p, "children": c})

        ctx["output"] = {"parents": total_parents, "children": total_children}

    return {"parents_ingested": total_parents, "children_ingested": total_children, "documents": results}


def ingest_texts(documents: list[tuple[str, str, str]]) -> dict:
    """documents: list of (text, source, category) tuples — raw text, no file involved."""
    settings = get_settings()
    ensure_schema()
    client = get_weaviate_client()
    encoder = get_encoder()
    parent_collection = client.collections.get(settings.WEAVIATE_CLASS_PARENT)
    child_collection = client.collections.get(settings.WEAVIATE_CLASS_CHILD)

    total_parents = total_children = 0
    results = []
    with log_step("ingest_texts", agent="rag_agent", num_documents=len(documents)) as ctx:
        for text, source, category in documents:
            content_hash = _hash_content(text)
            action = _resolve_action(source, content_hash)
            if action == "unchanged":
                results.append({"source": source, "status": "unchanged", "parents": 0, "children": 0})
                continue
            if action == "updated":
                deactivate_source(source)

            chunked = chunk_document(text, source, category)
            p, c = _upsert_chunked_document(chunked, content_hash, encoder, parent_collection, child_collection)
            total_parents += p
            total_children += c
            results.append({"source": source, "status": action, "parents": p, "children": c})

        ctx["output"] = {"parents": total_parents, "children": total_children}

    return {"parents_ingested": total_parents, "children_ingested": total_children, "documents": results}
