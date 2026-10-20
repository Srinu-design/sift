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
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_thinking_budget: int | None = None
    stub_delay_ms: int = Field(default=0, ge=0)

    # LLM price table (USD per 1M tokens): free tier costs 0.0; set paid rates and date checked.
    llm_price_in_per_mtok: float = 0.0
    llm_price_out_per_mtok: float = 0.0
    llm_prices_checked_on: str = ""

    # shortcut: one dev tenant, no auth; API keys and tenancy replace this in week 5.
    dev_tenant: str = "pgdocs"
    dev_collection: str = "pgdocs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
