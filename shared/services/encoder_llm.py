"""
Bi-encoder embedding service — used for dense vectors in the hybrid
retriever and for ingestion. Provider-selectable the same way as chat_llm.
"""
from functools import lru_cache

from shared.core.config import get_settings


class EmbeddingClient:
    """Thin uniform wrapper so callers don't care which backend is used."""

    def __init__(self, backend, kind: str):
        self._backend = backend
        self._kind = kind  # "langchain" | "sentence_transformers"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._kind == "sentence_transformers":
            return self._backend.encode(texts, normalize_embeddings=True).tolist()
        return self._backend.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._kind == "sentence_transformers":
            return self._backend.encode([text], normalize_embeddings=True)[0].tolist()
        return self._backend.embed_query(text)


@lru_cache
def get_encoder() -> EmbeddingClient:
    settings = get_settings()
    provider = settings.EMBEDDING_PROVIDER

    if provider == "sentence_transformers":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return EmbeddingClient(model, kind="sentence_transformers")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        model = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_API_KEY or settings.OPENAI_API_KEY,
        )
        return EmbeddingClient(model, kind="langchain")

    if provider == "local":
        from langchain_openai import OpenAIEmbeddings
        model = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            base_url=settings.LOCAL_LLM_BASE_URL,
            api_key=settings.LOCAL_LLM_API_KEY,
        )
        return EmbeddingClient(model, kind="langchain")

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")


@lru_cache
def get_reranker():
    """Cross-encoder reranker used as the final filter stage in retrieval."""
    from sentence_transformers import CrossEncoder
    settings = get_settings()
    return CrossEncoder(settings.RERANKER_MODEL)
