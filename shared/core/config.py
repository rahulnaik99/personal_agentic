"""
Central application settings.

All configuration is sourced from environment variables / a .env file.
Nothing here should be hardcoded — this module is the single source of
truth for provider selection, model params, and infra endpoints so that
agents never read os.environ directly.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- App ----
    APP_NAME: str = "agentic-langgraph-app"
    ENV: Literal["dev", "staging", "prod"] = "dev"
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True

    # ---- LLM provider selection (multi-provider) ----
    # "openai" | "anthropic" | "local"
    LLM_PROVIDER: Literal["openai", "anthropic", "local", "mlx"] = "anthropic"
    LLM_MODEL: str = "claude-sonnet-4-6"

    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL_FALLBACK: str = "gpt-4o-mini"

    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL_FALLBACK: str = "claude-sonnet-4-6"

    # Local model server (e.g. Ollama / vLLM OpenAI-compatible endpoint)
    LOCAL_LLM_BASE_URL: str = "http://localhost:11434/v1"
    LOCAL_LLM_API_KEY: str = "not-needed"
    LOCAL_MODEL_FALLBACK: str = "llama3.1"

    # MLX-LM OpenAI-compatible server on Apple Silicon.
    # Start with: mlx_lm.server --model <HF-MLX-model> --port 8080
    MLX_LLM_BASE_URL: str = "http://127.0.0.1:8080/v1"
    MLX_LLM_API_KEY: str = "not-needed"
    MLX_MODEL_FALLBACK: str = "mlx-community/DeepSeek-R1-Distill-Qwen-14B-MLX"

    # ---- Generation params (used unless a node overrides them) ----
    TEMPERATURE: float = 0.2
    TOP_P: float = 0.9
    TOP_K: int = 5              # note: also reused as default retrieval top_k below
    MAX_TOKENS: int = 1024

    # ---- Embeddings (bi-encoder) ----
    EMBEDDING_PROVIDER: Literal["openai", "anthropic", "local", "sentence_transformers"] = "sentence_transformers"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_API_KEY: str | None = None
    EMBEDDING_DIM: int = 384

    # ---- Reranker (cross-encoder filter stage) ----
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_N: int = 5

    # ---- Retrieval tuning ----
    RETRIEVAL_TOP_K_DENSE: int = 20
    RETRIEVAL_TOP_K_SPARSE: int = 20
    RRF_K: int = 60                 # RRF damping constant
    MMR_LAMBDA: float = 0.5         # 1.0 = pure relevance, 0.0 = pure diversity
    MMR_TOP_K: int = 10
    FINAL_TOP_K: int = 5            # after rerank, sent to generation

    # ---- Parent-child chunking ----
    CHILD_CHUNK_SIZE: int = 400
    CHILD_CHUNK_OVERLAP: int = 50
    PARENT_CHUNK_SIZE: int = 2000
    PARENT_CHUNK_OVERLAP: int = 200

    # ---- Weaviate (local via docker-compose) ----
    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_GRPC_URL: str = "localhost:50051"
    WEAVIATE_API_KEY: str | None = None  # unused for local, kept for parity
    WEAVIATE_CLASS_CHILD: str = "ChildChunk"
    WEAVIATE_CLASS_PARENT: str = "ParentChunk"

    # ---- Tool agent ----
    WEB_SEARCH_PROVIDER: Literal["duckduckgo"] = "duckduckgo"
    WEB_SEARCH_MAX_RESULTS: int = 5
    URL_FETCH_TIMEOUT_SECONDS: int = 15
    URL_FETCH_MAX_CHARS: int = 8000
    TOOL_ALLOWED_DOMAINS: str | None = None  # comma-separated allow-list, None = no restriction

    # ---- RAGAS evaluation ----
    RAGAS_ENABLED: bool = True
    RAGAS_LLM_MODEL: str | None = None  # falls back to LLM_MODEL if unset

    # ---- API (orchestrator's external HTTP/SSE gateway) ----
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "*"

    # ---- gRPC service endpoints (A2A agent services) ----
    RAG_AGENT_GRPC_HOST: str = "localhost"
    RAG_AGENT_GRPC_PORT: int = 50061
    # The RAG agent's separate HTTP ingest API (see rag_agent_service/app/ingest_api.py).
    # Used by the orchestrator's /status endpoint to check Weaviate connectivity
    # (the RAG agent is the only service that talks to Weaviate directly).
    RAG_AGENT_INGEST_URL: str = "http://localhost:8001"
    TOOL_AGENT_GRPC_HOST: str = "localhost"
    TOOL_AGENT_GRPC_PORT: int = 50062
    GRPC_CALL_TIMEOUT_SECONDS: int = 60

    # ---- Guardrails ----
    GUARDRAILS_ENABLED: bool = True
    GUARDRAILS_MAX_INPUT_CHARS: int = 4000
    GUARDRAILS_BLOCK_ON_PII_INPUT: bool = True
    GUARDRAILS_BLOCK_ON_PII_OUTPUT: bool = False  # redact instead of hard-block by default
    GUARDRAILS_LOW_FAITHFULNESS_THRESHOLD: float = 0.5  # below this, answer is flagged low-confidence

    # ---- PDF/document ingestion (unstructured) ----
    UNSTRUCTURED_PDF_STRATEGY: str = "hi_res"  # "fast" | "hi_res" | "ocr_only"
    INGESTION_IMAGE_DIR: str = "./data/ingested_images"
    VISION_CAPTION_MODEL_PROVIDER: Literal["openai", "anthropic"] = "anthropic"

    # ---- RAGAS regression benchmark ----
    RAGAS_REGRESSION_TOLERANCE: float = 0.03  # max allowed drop vs baseline per metric


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor — import and call this, never instantiate Settings() directly."""
    return Settings()
