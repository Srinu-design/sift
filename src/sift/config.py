from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process settings. Environment variables only; no config files."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    # One Postgres (the pgvector image) holds Hopper's database and this one, "sift".
    database_url: str = "postgresql+asyncpg://hopper:hopper@127.0.0.1:5432/sift"
    log_level: str = "INFO"
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=0, ge=0)

    # Models, pinned to a Hugging Face commit so an eval number always names what produced it.
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_revision: str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    # sift-models: api and ingest call it for /embed and /rerank.
    models_url: str = "http://127.0.0.1:8100"
    models_timeout_seconds: float = Field(default=60.0, gt=0)

    # Raw uploads, stored under <tenant id>/<sha256>. A local folder in dev, S3 in production.
    file_store: str = "data/files"

    # Hopper runs ingestion. Sift is a tenant there: its API key enqueues jobs, and its
    # signing secret (base64, as Hopper shows it once) verifies the jobs' calls to sift-ingest.
    hopper_url: str = "http://127.0.0.1:8000"
    hopper_api_key: SecretStr = SecretStr("")
    hopper_signing_secret: SecretStr = SecretStr("")
    # How Hopper's workers reach sift-ingest: a Docker network name, allowlisted in Hopper.
    ingest_url: str = "http://sift-ingest:8000"
    # Hopper-Signature timestamps older (or newer) than this are rejected as replays.
    signature_tolerance_seconds: int = Field(default=300, ge=1)

    # Rerank the fused top 30: same dev quality as 50 at 60% of the latency (ADR-0003).
    rerank_depth: int = 30
    context_k: int = 8
    # Gate 1: refuse below this best rerank score. `make eval-tau` on dev, 2026-10-20: the
    # largest tau that wrongly refuses at most 5% of answerable questions (ADR-0008).
    refusal_tau: float = 0.19

    # Answer LLM settings (Gemini over REST, or deterministic stub for tests and load tests).
    llm_provider: Literal["gemini", "stub"] = "stub"
    gemini_api_key: SecretStr = SecretStr("")
    gemini_url: str = "https://generativelanguage.googleapis.com"
    # Checked 2026-10-20 on the free tier: 2.5-flash is closed to new keys, 3.8-flash hung past
    # 150 s, 3.5-flash answered 503; 3.5-flash-lite returned valid cited JSON in 1.6 s.
    answer_model: str = "gemini-3.5-flash-lite"
    # The judge must be a different model from the answerer, validated against hand labels
    # (kappa). Checked 2026-10-23 on the free tier: pro models have no quota, 3.5-flash allows 20
    # requests a day, Gemma 4 31B kept failing with 500/503; 3.1-flash-lite (an older version of
    # the answerer's family, so not stronger) answers the judge schema reliably.
    judge_model: str = "gemini-3.1-flash-lite"
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_thinking_budget: int | None = None
    stub_delay_ms: int = Field(default=0, ge=0)

    # LLM price table (USD per 1M tokens): free tier costs 0.0; set paid rates and date checked.
    llm_price_in_per_mtok: float = 0.0
    llm_price_out_per_mtok: float = 0.0
    llm_prices_checked_on: str = ""

    # Tenancy. API keys are stored as HMAC-SHA256(pepper, secret); without the pepper a leaked
    # table cannot be checked against. The API runs as sift_app, so row-level security applies.
    api_key_pepper: SecretStr = SecretStr("")
    api_db_role: str | None = "sift_app"
    max_upload_bytes: int = Field(default=20_000_000, ge=1)
    # The public demo page queries this tenant without a key; None turns /demo off.
    demo_tenant: str | None = "pgdocs"
    demo_collection: str = "pgdocs"
    demo_rate_per_minute: int = Field(default=10, ge=1)  # per client IP

    # Ops (week 6). Redis holds the query-embedding and answer caches, the rate limiter's
    # buckets and each tenant's daily spend; every use fails open while Redis is down.
    redis_url: str = "redis://127.0.0.1:6379/1"  # db 1: Hopper uses db 0
    redis_retry_seconds: float = Field(default=5.0, gt=0)
    metrics_port: int = Field(default=9100, ge=0, le=65535)  # 0 serves nothing (tests)
    query_embedding_ttl_seconds: int = Field(default=86_400, ge=1)
    answer_cache_enabled: bool = True
    answer_cache_ttl_seconds: int = Field(default=86_400, ge=1)
    query_rate_per_sec: float = Field(default=2.0, gt=0)  # per tenant, /v1/query
    query_burst: int = Field(default=10, ge=1)
    # Per-tenant daily LLM spend; over it, /v1/query answers 429 until UTC midnight. None: no cap.
    daily_budget_usd: float | None = Field(default=1.0, gt=0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
