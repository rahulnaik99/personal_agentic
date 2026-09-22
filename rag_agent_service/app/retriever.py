"""
Hybrid retriever pipeline (dense + sparse -> manual RRF -> MMR ->
cross-encoder rerank -> parent expansion), now carrying element_type
through so generation can render tables/images differently from prose.
"""
from __future__ import annotations

from dataclasses import dataclass

import json

import numpy as np
from weaviate.classes.query import Filter

from shared.core.config import get_settings
from shared.core.logging import log_step
from shared.services.encoder_llm import get_encoder, get_reranker
from shared.vectorstore.weaviate_client import (
    _graphql_escape,
    _graphql_request,
    get_weaviate_client,
)


def _active_filter(category: str | None):
    """Build the non-boolean part of the client filter.

    Weaviate Python client 4.7.1 has a boolean-filter serialization bug.
    Dense/sparse retrieval therefore uses GraphQL below, where
    ``valueBoolean: true`` is explicit.
    """
    if category:
        return Filter.by_property("category").equal(category)
    return None


def _where_clause(category: str | None) -> str:
    """Return a GraphQL where expression that always enforces is_active=true."""
    active = '''
      path: ["is_active"]
      operator: Equal
      valueBoolean: true
    '''

    if not category:
        return "{ " + active + " }"

    category_value = _graphql_escape(category)
    return f'''{{
      operator: And
      operands: [
        {{ {active} }},
        {{
          path: ["category"]
          operator: Equal
          valueText: "{category_value}"
        }}
      ]
    }}'''


def _dense_search_graphql(
    query_vector: list[float],
    top_k: int,
    category: str | None,
    child_class: str,
) -> list[RetrievedChunk]:
    vector_literal = json.dumps(
        [float(v) for v in query_vector],
        separators=(",", ":"),
    )
    where = _where_clause(category)

    gql = f"""
    {{
      Get {{
        {child_class}(
          nearVector: {{ vector: {vector_literal} }}
          where: {where}
          limit: {int(top_k)}
        ) {{
          text
          parent_id
          source
          element_type
          _additional {{
            id
            distance
          }}
        }}
      }}
    }}
    """

    data = _graphql_request(gql)
    rows = data.get("Get", {}).get(child_class, [])

    out: list[RetrievedChunk] = []
    for obj in rows:
        additional = obj.get("_additional") or {}
        distance = float(additional.get("distance") or 1.0)
        out.append(
            RetrievedChunk(
                id=str(additional.get("id")),
                text=obj.get("text", ""),
                parent_id=obj.get("parent_id", ""),
                source=obj.get("source", ""),
                score=1.0 - distance,
                element_type=obj.get("element_type", "text"),
            )
        )
    return out


def _sparse_search_graphql(
    query: str,
    top_k: int,
    category: str | None,
    child_class: str,
) -> list[RetrievedChunk]:
    query_value = _graphql_escape(query)
    where = _where_clause(category)

    gql = f"""
    {{
      Get {{
        {child_class}(
          bm25: {{ query: "{query_value}" }}
          where: {where}
          limit: {int(top_k)}
        ) {{
          text
          parent_id
          source
          element_type
          _additional {{
            id
            score
          }}
        }}
      }}
    }}
    """

    data = _graphql_request(gql)
    rows = data.get("Get", {}).get(child_class, [])

    out: list[RetrievedChunk] = []
    for obj in rows:
        additional = obj.get("_additional") or {}
        out.append(
            RetrievedChunk(
                id=str(additional.get("id")),
                text=obj.get("text", ""),
                parent_id=obj.get("parent_id", ""),
                source=obj.get("source", ""),
                score=float(additional.get("score") or 0.0),
                element_type=obj.get("element_type", "text"),
            )
        )
    return out


@dataclass
class RetrievedChunk:
    id: str
    text: str
    parent_id: str
    source: str
    score: float
    element_type: str = "text"


@dataclass
class RetrievedContext:
    parent_id: str
    text: str
    source: str
    element_type: str = "text"


def _dense_search(query: str, top_k: int, category: str | None) -> list[RetrievedChunk]:
    settings = get_settings()
    encoder = get_encoder()
    query_vector = encoder.embed_query(query)

    # GraphQL explicitly encodes BOOL as valueBoolean, avoiding the
    # weaviate-client 4.7.1 gRPC bool-filter serialization bug.
    return _dense_search_graphql(
        query_vector=query_vector,
        top_k=top_k,
        category=category,
        child_class=settings.WEAVIATE_CLASS_CHILD,
    )


def _sparse_search(query: str, top_k: int, category: str | None) -> list[RetrievedChunk]:
    settings = get_settings()

    return _sparse_search_graphql(
        query=query,
        top_k=top_k,
        category=category,
        child_class=settings.WEAVIATE_CLASS_CHILD,
    )


def _reciprocal_rank_fusion(ranked_lists: list[list[RetrievedChunk]], k: int) -> list[RetrievedChunk]:
    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            chunk_lookup[chunk.id] = chunk
    fused = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        RetrievedChunk(
            id=chunk_lookup[cid].id, text=chunk_lookup[cid].text, parent_id=chunk_lookup[cid].parent_id,
            source=chunk_lookup[cid].source, score=score, element_type=chunk_lookup[cid].element_type,
        )
        for cid, score in fused
    ]


def _mmr(query_vector: list[float], candidates: list[RetrievedChunk], lambda_mult: float, top_k: int) -> list[RetrievedChunk]:
    if not candidates:
        return []
    encoder = get_encoder()
    candidate_vectors = np.array(encoder.embed_documents([c.text for c in candidates]))
    query_vec = np.array(query_vector)

    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
        return float(np.dot(a, b) / denom)

    relevance = [cos_sim(query_vec, v) for v in candidate_vectors]
    selected_idx: list[int] = []
    remaining_idx = list(range(len(candidates)))

    while remaining_idx and len(selected_idx) < top_k:
        if not selected_idx:
            best = max(remaining_idx, key=lambda i: relevance[i])
        else:
            def mmr_score(i: int) -> float:
                diversity_penalty = max(cos_sim(candidate_vectors[i], candidate_vectors[j]) for j in selected_idx)
                return lambda_mult * relevance[i] - (1 - lambda_mult) * diversity_penalty
            best = max(remaining_idx, key=mmr_score)
        selected_idx.append(best)
        remaining_idx.remove(best)

    return [candidates[i] for i in selected_idx]


def _cross_encoder_rerank(query: str, candidates: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    if not candidates:
        return []
    reranker = get_reranker()
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
    return [c for c, _ in ranked[:top_n]]


def _expand_to_parents(chunks: list[RetrievedChunk]) -> list[RetrievedContext]:
    settings = get_settings()
    client = get_weaviate_client()
    parent_collection = client.collections.get(settings.WEAVIATE_CLASS_PARENT)

    seen_parent_ids: set[str] = set()
    contexts: list[RetrievedContext] = []
    for chunk in chunks:
        if chunk.parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(chunk.parent_id)
        parent_obj = parent_collection.query.fetch_object_by_id(chunk.parent_id)
        if parent_obj:
            contexts.append(RetrievedContext(
                parent_id=chunk.parent_id, text=parent_obj.properties["text"],
                source=parent_obj.properties["source"], element_type=chunk.element_type,
            ))
    return contexts


def hybrid_retrieve(query: str, category: str | None = None) -> list[RetrievedContext]:
    settings = get_settings()
    encoder = get_encoder()

    with log_step("hybrid_retrieve", agent="rag_agent", query=query, category=category) as ctx:
        dense_hits = _dense_search(query, settings.RETRIEVAL_TOP_K_DENSE, category)
        sparse_hits = _sparse_search(query, settings.RETRIEVAL_TOP_K_SPARSE, category)
        fused = _reciprocal_rank_fusion([dense_hits, sparse_hits], k=settings.RRF_K)

        query_vector = encoder.embed_query(query)
        diversified = _mmr(query_vector, fused, settings.MMR_LAMBDA, settings.MMR_TOP_K)
        reranked = _cross_encoder_rerank(query, diversified, settings.FINAL_TOP_K)
        contexts = _expand_to_parents(reranked)
        ctx["output"] = {"num_contexts": len(contexts), "category": category}

    return contexts
