"""
Pure-logic tests for the retriever's fusion/diversification math — no
Weaviate, no LLM, no real embeddings. get_encoder() is monkeypatched to
a deterministic fake so MMR selection order is fully predictable.
"""

from rag_agent_service.app.retriever import RetrievedChunk, _mmr, _reciprocal_rank_fusion


def make_chunk(cid: str, text: str = "text") -> RetrievedChunk:
    return RetrievedChunk(id=cid, text=text, parent_id=f"parent-{cid}", source="test", score=0.0)


class FakeEncoder:
    """Returns fixed, hand-picked vectors so relevance/diversity are deterministic."""
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def embed_documents(self, texts):
        return [self.vectors[t] for t in texts]

    def embed_query(self, text):
        return self.vectors.get(text, [1.0, 0.0])


def test_rrf_favors_items_ranked_high_in_both_lists():
    a, b, c = make_chunk("a"), make_chunk("b"), make_chunk("c")
    dense = [a, b, c]     # a=rank0, b=rank1, c=rank2
    sparse = [b, a, c]    # b=rank0, a=rank1, c=rank2

    fused = _reciprocal_rank_fusion([dense, sparse], k=60)
    fused_ids = [f.id for f in fused]

    # a and b are both ranked in the top 2 across both lists; c is last in both.
    assert fused_ids[-1] == "c"
    assert set(fused_ids[:2]) == {"a", "b"}


def test_rrf_handles_disjoint_lists():
    a, b = make_chunk("a"), make_chunk("b")
    fused = _reciprocal_rank_fusion([[a], [b]], k=60)
    assert {f.id for f in fused} == {"a", "b"}


def test_rrf_empty_lists_returns_empty():
    assert _reciprocal_rank_fusion([[], []], k=60) == []


def test_mmr_selects_most_relevant_first(monkeypatch):
    # "near" is very close to the query vector; "far" is orthogonal.
    near = make_chunk("near", text="near")
    far = make_chunk("far", text="far")

    fake_vectors = {"near": [1.0, 0.0], "far": [0.0, 1.0]}
    monkeypatch.setattr(
        "rag_agent_service.app.retriever.get_encoder",
        lambda: FakeEncoder(fake_vectors),
    )

    selected = _mmr(query_vector=[1.0, 0.0], candidates=[far, near], lambda_mult=1.0, top_k=1)
    assert selected[0].id == "near"


def test_mmr_reduces_redundancy_with_low_lambda(monkeypatch):
    # Two near-duplicate candidates plus one distinct one; with lambda
    # skewed toward diversity, MMR should not pick both duplicates first.
    dup1 = make_chunk("dup1", text="dup1")
    dup2 = make_chunk("dup2", text="dup2")
    distinct = make_chunk("distinct", text="distinct")

    fake_vectors = {
        "dup1": [1.0, 0.0],
        "dup2": [0.99, 0.01],   # nearly identical to dup1
        "distinct": [0.0, 1.0],
    }
    monkeypatch.setattr(
        "rag_agent_service.app.retriever.get_encoder",
        lambda: FakeEncoder(fake_vectors),
    )

    selected = _mmr(
        query_vector=[1.0, 0.0], candidates=[dup1, dup2, distinct],
        lambda_mult=0.3, top_k=2,
    )
    selected_ids = {c.id for c in selected}
    # dup1 wins on pure relevance first; the diversity term should then
    # favor "distinct" over the near-duplicate "dup2" for the 2nd pick.
    assert "dup1" in selected_ids
    assert "distinct" in selected_ids


def test_mmr_empty_candidates_returns_empty():
    assert _mmr(query_vector=[1.0, 0.0], candidates=[], lambda_mult=0.5, top_k=5) == []
