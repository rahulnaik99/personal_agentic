"""
Weaviate connection + schema management.

Two collections:
  - ParentChunk: large context blocks, stored for retrieval-time expansion,
    NOT directly vector-searched.
  - ChildChunk: small chunks, vector-searched and text-searched.

Dense and sparse retrieval are intentionally performed separately and fused
manually in the RAG retriever.

IMPORTANT:
    weaviate-client==4.7.1 has a Boolean-filter serialization problem in
    query.fetch_objects().

    The schema intentionally keeps:
        is_active = BOOL
        category  = TEXT

    Boolean-filtered reads therefore use Weaviate's HTTP GraphQL API with
    explicit valueBoolean:true/false.

    Writes continue to use the official Weaviate Python client.
"""

from datetime import UTC, datetime
from functools import lru_cache

import httpx
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter

from shared.core.config import get_settings


# ============================================================================
# Weaviate connection
# ============================================================================

@lru_cache
def get_weaviate_client() -> weaviate.WeaviateClient:
    settings = get_settings()

    client = weaviate.connect_to_local(
        host=_host_from_url(settings.WEAVIATE_URL),
        port=_port_from_url(settings.WEAVIATE_URL),
        grpc_port=int(
            settings.WEAVIATE_GRPC_URL.split(":")[-1]
        ),
    )

    return client


def _host_from_url(url: str) -> str:
    return (
        url.split("//")[-1]
        .split(":")[0]
    )


def _port_from_url(url: str) -> int:
    return int(
        url.split("//")[-1]
        .split(":")[1]
    )


# ============================================================================
# GraphQL helpers
# ============================================================================

def _graphql_escape(value: str) -> str:
    """
    Escape a Python string before embedding it in a GraphQL string literal.

    GraphQL strings use JSON-like escaping rules.
    """

    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _graphql_request(query: str) -> dict:
    """
    Execute a GraphQL request against Weaviate.

    We intentionally use HTTP GraphQL for Boolean-filtered reads because
    weaviate-client==4.7.1 has a gRPC protobuf serialization issue where:

        valueBoolean=True

    can incorrectly be passed into:

        value_int

    GraphQL lets us explicitly send:

        valueBoolean: true
    """

    settings = get_settings()

    graphql_url = (
        f"{settings.WEAVIATE_URL.rstrip('/')}"
        "/v1/graphql"
    )

    response = httpx.post(
        graphql_url,
        json={"query": query},
        timeout=30.0,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("errors"):
        raise RuntimeError(
            "Weaviate GraphQL error: "
            f"{payload['errors']}"
        )

    return payload.get("data", {})


# ============================================================================
# Shared schema properties
# ============================================================================

_VERSIONING_PROPERTIES = [
    Property(
        name="category",
        data_type=DataType.TEXT,
    ),
    Property(
        name="is_active",
        data_type=DataType.BOOL,
    ),
    Property(
        name="ingested_at",
        data_type=DataType.DATE,
    ),
    Property(
        name="content_hash",
        data_type=DataType.TEXT,
    ),
]


# ============================================================================
# Schema management
# ============================================================================

def ensure_schema() -> None:
    """
    Idempotently create the Parent/Child collections if missing.
    """

    settings = get_settings()
    client = get_weaviate_client()

    # ------------------------------------------------------------------------
    # ParentChunk
    # ------------------------------------------------------------------------

    if not client.collections.exists(
        settings.WEAVIATE_CLASS_PARENT
    ):
        client.collections.create(
            name=settings.WEAVIATE_CLASS_PARENT,
            properties=[
                Property(
                    name="text",
                    data_type=DataType.TEXT,
                ),
                Property(
                    name="source",
                    data_type=DataType.TEXT,
                ),
                Property(
                    name="metadata_json",
                    data_type=DataType.TEXT,
                ),
                *_VERSIONING_PROPERTIES,
            ],
            vectorizer_config=Configure.Vectorizer.none(),
        )

    # ------------------------------------------------------------------------
    # ChildChunk
    # ------------------------------------------------------------------------

    if not client.collections.exists(
        settings.WEAVIATE_CLASS_CHILD
    ):
        client.collections.create(
            name=settings.WEAVIATE_CLASS_CHILD,
            properties=[
                Property(
                    name="text",
                    data_type=DataType.TEXT,
                ),
                Property(
                    name="parent_id",
                    data_type=DataType.TEXT,
                ),
                Property(
                    name="source",
                    data_type=DataType.TEXT,
                ),
                Property(
                    name="chunk_index",
                    data_type=DataType.INT,
                ),
                Property(
                    name="element_type",
                    data_type=DataType.TEXT,
                ),
                Property(
                    name="image_path",
                    data_type=DataType.TEXT,
                ),
                *_VERSIONING_PROPERTIES,
            ],
            vectorizer_config=Configure.Vectorizer.none(),
        )


# ============================================================================
# Existing application queries
# ============================================================================

def get_active_categories() -> list[str]:
    """
    Return distinct categories currently present among active ChildChunk
    objects.

    Uses GraphQL because is_active is a BOOL and weaviate-client 4.7.1
    fetch_objects() has a Boolean filter serialization issue.
    """

    settings = get_settings()

    child_class = settings.WEAVIATE_CLASS_CHILD

    # In Weaviate GraphQL, ``groupBy`` is an argument on the
    # collection aggregation, while ``groupedBy`` is the response field.
    query = f"""
    {{
      Aggregate {{
        {child_class}(
          where: {{
            path: ["is_active"]
            operator: Equal
            valueBoolean: true
          }}
          groupBy: ["category"]
        ) {{
          groupedBy {{
            value
          }}
        }}
      }}
    }}
    """


    data = _graphql_request(query)

    rows = (
        data
        .get("Aggregate", {})
        .get(child_class, [])
    )

    categories: set[str] = set()

    for row in rows:
        grouped_by = row.get("groupedBy") or {}
        value = grouped_by.get("value")

        if value:
            categories.add(value)

    return sorted(categories)


# ============================================================================
# Duplicate detection
# ============================================================================

def get_existing_chunk_hash(
    source: str,
) -> str | None:
    """
    Return the content_hash of the currently-active ParentChunk for a source.

    IMPORTANT:
        Do NOT use:

            collection.query.fetch_objects(
                filters=Filter.by_property(
                    "is_active"
                ).equal(True)
            )

        with weaviate-client==4.7.1.

    Instead, Boolean filtering is performed through GraphQL.
    """

    settings = get_settings()

    parent_class = settings.WEAVIATE_CLASS_PARENT

    escaped_source = _graphql_escape(source)

    query = f"""
    {{
      Get {{
        {parent_class}(
          where: {{
            operator: And
            operands: [
              {{
                path: ["source"]
                operator: Equal
                valueText: "{escaped_source}"
              }}
              {{
                path: ["is_active"]
                operator: Equal
                valueBoolean: true
              }}
            ]
          }}
          limit: 1
        ) {{
          source
          content_hash
          is_active
        }}
      }}
    }}
    """

    data = _graphql_request(query)

    objects = (
        data
        .get("Get", {})
        .get(parent_class, [])
    )

    if not objects:
        return None

    return objects[0].get("content_hash")


# ============================================================================
# Source deactivation
# ============================================================================

def _get_active_objects_by_source(
    collection_name: str,
    source: str,
) -> list[dict]:
    """
    Return active objects matching a source.

    Only the properties needed for deactivation are retrieved.

    GraphQL is used to avoid the weaviate-client 4.7.1 Boolean-filter bug.
    """

    escaped_source = _graphql_escape(source)

    query = f"""
    {{
      Get {{
        {collection_name}(
          where: {{
            operator: And
            operands: [
              {{
                path: ["source"]
                operator: Equal
                valueText: "{escaped_source}"
              }}
              {{
                path: ["is_active"]
                operator: Equal
                valueBoolean: true
              }}
            ]
          }}
          limit: 10000
        ) {{
          _additional {{
            id
          }}
          source
          is_active
        }}
      }}
    }}
    """

    data = _graphql_request(query)

    return (
        data
        .get("Get", {})
        .get(collection_name, [])
    )


def deactivate_source(
    source: str,
) -> int:
    """
    Mark every active parent/child chunk for a source as inactive.

    Reads are performed through GraphQL because of the Boolean-filter issue
    in weaviate-client 4.7.1.

    Updates are still performed using the normal Weaviate Python client.
    """

    settings = get_settings()
    client = get_weaviate_client()

    deactivated = 0

    for class_name in (
        settings.WEAVIATE_CLASS_PARENT,
        settings.WEAVIATE_CLASS_CHILD,
    ):

        collection = client.collections.get(
            class_name
        )

        objects = _get_active_objects_by_source(
            class_name,
            source,
        )

        for obj in objects:

            additional = (
                obj.get("_additional")
                or {}
            )

            uuid = additional.get("id")

            if not uuid:
                continue

            collection.data.update(
                uuid=uuid,
                properties={
                    "is_active": False,
                },
            )

            deactivated += 1

    return deactivated


# ============================================================================
# Dashboard statistics
# ============================================================================

def _graphql_count(
    collection_name: str,
    is_active: bool,
) -> int:
    """
    Count objects using an explicit GraphQL Boolean filter.
    """

    value = (
        "true"
        if is_active
        else "false"
    )

    query = f"""
    {{
      Aggregate {{
        {collection_name}(
          where: {{
            path: ["is_active"]
            operator: Equal
            valueBoolean: {value}
          }}
        ) {{
          meta {{
            count
          }}
        }}
      }}
    }}
    """

    data = _graphql_request(query)

    rows = (
        data
        .get("Aggregate", {})
        .get(collection_name, [])
    )

    if not rows:
        return 0

    return (
        rows[0]
        .get("meta", {})
        .get("count", 0)
        or 0
    )


def _get_active_parent_sources() -> list[dict]:
    """
    Fetch active ParentChunk category/source pairs.

    Uses HTTP GraphQL instead of client 4.7.1's gRPC fetch_objects Boolean
    filter serialization path.
    """

    settings = get_settings()
    parent_class = settings.WEAVIATE_CLASS_PARENT

    query = f"""
    {{
      Get {{
        {parent_class}(
          where: {{
            path: ["is_active"]
            operator: Equal
            valueBoolean: true
          }}
          limit: 10000
        ) {{
          category
          source
        }}
      }}
    }}
    """

    data = _graphql_request(query)

    return (
        data
        .get("Get", {})
        .get(parent_class, [])
    )


def get_ingestion_stats() -> dict:
    """
    Powers the Streamlit ingestion dashboard.

    Returns:
      - total object count
      - active/inactive ChildChunk counts
      - active/inactive ParentChunk counts
      - active embedding count
      - distinct files by category
      - total distinct files

    is_active remains BOOL.
    category remains TEXT.
    """

    settings = get_settings()

    ensure_schema()

    # ------------------------------------------------------------------------
    # Child counts
    # ------------------------------------------------------------------------

    total_chunks_active = _graphql_count(
        settings.WEAVIATE_CLASS_CHILD,
        True,
    )

    total_chunks_inactive = _graphql_count(
        settings.WEAVIATE_CLASS_CHILD,
        False,
    )

    # ------------------------------------------------------------------------
    # Parent counts
    # ------------------------------------------------------------------------

    total_parents_active = _graphql_count(
        settings.WEAVIATE_CLASS_PARENT,
        True,
    )

    total_parents_inactive = _graphql_count(
        settings.WEAVIATE_CLASS_PARENT,
        False,
    )

    # ------------------------------------------------------------------------
    # Category -> distinct source/file count
    # ------------------------------------------------------------------------

    sources_by_category: dict[
        str,
        set[str],
    ] = {}

    active_parents = (
        _get_active_parent_sources()
    )

    for obj in active_parents:

        category = (
            obj.get("category")
            or "uncategorized"
        )

        source = obj.get("source")

        if source:

            sources_by_category.setdefault(
                category,
                set(),
            ).add(source)

    category_file_counts = {
        category: len(sources)
        for category, sources
        in sources_by_category.items()
    }

    # ------------------------------------------------------------------------
    # Final response
    # ------------------------------------------------------------------------

    return {
        "total_objects": (
            total_chunks_active
            + total_chunks_inactive
            + total_parents_active
            + total_parents_inactive
        ),
        "total_chunks_active": (
            total_chunks_active
        ),
        "total_chunks_inactive": (
            total_chunks_inactive
        ),
        "total_embeddings_active": (
            total_chunks_active
        ),
        "total_files_by_category": (
            category_file_counts
        ),
        "total_files": (
            sum(
                category_file_counts.values()
            )
        ),
    }


# ============================================================================
# Utilities
# ============================================================================

def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def close_client() -> None:
    """
    Close the cached Weaviate connection.
    """

    client = get_weaviate_client()
    client.close()