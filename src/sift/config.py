from functools import lru_cache

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
