"""
Weaviate connection + schema management.

Two collections:
  - ParentChunk: large context blocks, stored for retrieval-time expansion,
    NOT directly vector-searched (no need — we search children and expand).
  - ChildChunk:  small chunks, vector-searched (dense) and text-searched
    (BM25 sparse via Weaviate's inverted index), each carries a
    `parent_id` reference back to its ParentChunk.

We deliberately do NOT use Weaviate's built-in `hybrid()` fused query —
per project design, dense and sparse are queried separately here and
fused manually (RRF) in app/agents/rag_agent/retriever.py, so each stage
(RRF, MMR, rerank) stays independently tunable.
"""
from datetime import UTC, datetime
from functools import lru_cache

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter

from shared.core.config import get_settings


@lru_cache
def get_weaviate_client() -> weaviate.WeaviateClient:
    settings = get_settings()
    client = weaviate.connect_to_local(
        host=_host_from_url(settings.WEAVIATE_URL),
        port=_port_from_url(settings.WEAVIATE_URL),
        grpc_port=int(settings.WEAVIATE_GRPC_URL.split(":")[-1]),
    )
    return client


def _host_from_url(url: str) -> str:
    return url.split("//")[-1].split(":")[0]


def _port_from_url(url: str) -> int:
    return int(url.split("//")[-1].split(":")[1])


# Properties shared by both collections for versioning/dedup + categorization.
# Defined once so ParentChunk and ChildChunk can never drift out of sync.
_VERSIONING_PROPERTIES = [
    # User-supplied label (e.g. "profession_doc", "financial_doc") — lets
    # retrieval filter to the right document category instead of searching
    # everything. See rag_agent_service/app/agent.py's classify_category().
    Property(name="category", data_type=DataType.TEXT),
    # Only active chunks are ever returned by retrieval (see retriever.py's
    # is_active filter). Re-ingesting the same `source` with changed content
    # deactivates the old chunks rather than deleting them, so history is
    # recoverable — see ingestion.py's upsert logic.
    Property(name="is_active", data_type=DataType.BOOL),
    Property(name="ingested_at", data_type=DataType.DATE),
    # SHA-256 of the source content, used to tell "genuine re-ingest of
    # identical content" (skip) apart from "content actually changed"
    # (deactivate old, insert new) for the same `source` label.
    Property(name="content_hash", data_type=DataType.TEXT),
]


def ensure_schema() -> None:
    """Idempotently create the Parent/Child collections if missing."""
    settings = get_settings()
    client = get_weaviate_client()

    if not client.collections.exists(settings.WEAVIATE_CLASS_PARENT):
        client.collections.create(
            name=settings.WEAVIATE_CLASS_PARENT,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="metadata_json", data_type=DataType.TEXT),
                *_VERSIONING_PROPERTIES,
            ],
            vectorizer_config=Configure.Vectorizer.none(),  # we supply/skip vectors ourselves
        )

    if not client.collections.exists(settings.WEAVIATE_CLASS_CHILD):
        client.collections.create(
            name=settings.WEAVIATE_CLASS_CHILD,
            properties=[
                Property(name="text", data_type=DataType.TEXT),  # BM25 sparse search hits this
                Property(name="parent_id", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                # "text" | "table" | "image" — tables/images are kept as single
                # unsplit chunks (see rag_agent_service/app/chunking.py) so
                # generation can treat them specially (render tables verbatim,
                # etc.) instead of prose-summarizing them.
                Property(name="element_type", data_type=DataType.TEXT),
                Property(name="image_path", data_type=DataType.TEXT),  # set only for element_type="image"
                *_VERSIONING_PROPERTIES,
            ],
            vectorizer_config=Configure.Vectorizer.none(),  # we supply our own bi-encoder vectors
        )


def get_active_categories() -> list[str]:
    """Distinct categories currently present among active chunks — used by
    the category classifier prompt to know what it's choosing between."""
    settings = get_settings()
    client = get_weaviate_client()
    child_collection = client.collections.get(settings.WEAVIATE_CLASS_CHILD)

    result = child_collection.aggregate.over_all(
        filters=Filter.by_property("is_active").equal(True),
        group_by="category",
    )
    return sorted({group.grouped_by.value for group in result.groups if group.grouped_by.value})


def get_existing_chunk_hash(source: str) -> str | None:
    """Returns the content_hash of the currently-active chunks for a given
    `source`, or None if nothing active exists for it yet. Used by
    ingestion.py to decide: first ingest / unchanged duplicate / real update."""
    settings = get_settings()
    client = get_weaviate_client()
    parent_collection = client.collections.get(settings.WEAVIATE_CLASS_PARENT)

    result = parent_collection.query.fetch_objects(
        filters=Filter.by_property("source").equal(source) & Filter.by_property("is_active").equal(True),
        limit=1,
    )
    if not result.objects:
        return None
    return result.objects[0].properties.get("content_hash")


def deactivate_source(source: str) -> int:
    """Marks every active parent+child chunk for a given `source` as
    inactive (never deleted, so history stays recoverable). Returns the
    number of objects deactivated."""
    settings = get_settings()
    client = get_weaviate_client()
    deactivated = 0

    for class_name in (settings.WEAVIATE_CLASS_PARENT, settings.WEAVIATE_CLASS_CHILD):
        collection = client.collections.get(class_name)
        result = collection.query.fetch_objects(
            filters=Filter.by_property("source").equal(source) & Filter.by_property("is_active").equal(True),
            limit=10_000,
        )
        for obj in result.objects:
            collection.data.update(uuid=obj.uuid, properties={"is_active": False})
            deactivated += 1

    return deactivated


def get_ingestion_stats() -> dict:
    """
    Powers the Streamlit ingestion dashboard: total object counts, a
    category -> distinct-file-count breakdown, and active/inactive chunk
    counts. Aggregate counts use Weaviate's native aggregate API (cheap);
    the per-category file breakdown fetches active parent objects and
    counts distinct `source` values in Python, since Weaviate's group-by
    aggregate counts objects, not distinct property values — fine at the
    scale this project targets, but would need a different approach
    (e.g. a maintained counter) at very large corpus sizes.
    """
    settings = get_settings()
    client = get_weaviate_client()
    parent_collection = client.collections.get(settings.WEAVIATE_CLASS_PARENT)
    child_collection = client.collections.get(settings.WEAVIATE_CLASS_CHILD)

    def _count(collection, is_active: bool) -> int:
        result = collection.aggregate.over_all(
            filters=Filter.by_property("is_active").equal(is_active), total_count=True,
        )
        return result.total_count or 0

    total_chunks_active = _count(child_collection, True)
    total_chunks_inactive = _count(child_collection, False)
    total_parents_active = _count(parent_collection, True)
    total_parents_inactive = _count(parent_collection, False)

    sources_by_category: dict[str, set] = {}
    result = parent_collection.query.fetch_objects(
        filters=Filter.by_property("is_active").equal(True),
        limit=10_000,
        return_properties=["category", "source"],
    )
    for obj in result.objects:
        category = obj.properties.get("category") or "uncategorized"
        source = obj.properties.get("source")
        if source:
            sources_by_category.setdefault(category, set()).add(source)

    category_file_counts = {category: len(sources) for category, sources in sources_by_category.items()}

    return {
        "total_objects": total_chunks_active + total_chunks_inactive + total_parents_active + total_parents_inactive,
        "total_chunks_active": total_chunks_active,
        "total_chunks_inactive": total_chunks_inactive,
        "total_embeddings_active": total_chunks_active,  # every active child chunk carries a vector
        "total_files_by_category": category_file_counts,
        "total_files": sum(category_file_counts.values()),
    }


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def close_client() -> None:
    client = get_weaviate_client()
    client.close()
