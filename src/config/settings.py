from __future__ import annotations
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = Field(default="DEV", alias="MODE")
    app_name: str = Field(default="Jojoba Travel Dashboard", alias="APP_NAME")
    app_version: str = Field(default="v1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    api_host: str = Field(default="0.0.0.0", alias="HOST")
    api_port: int = Field(default=8001, alias="PORT")

    mongo_url_dev: str = Field(default="mongodb://127.0.0.1:27017", alias="MONGO_URL")
    mongo_url_prod: str = Field(default="mongodb://127.0.0.1:27017", alias="MONGO_URL_2")
    database_name: str = Field(default="", alias="DATABASE_NAME")
    
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_MODEL")
    

    embedding_model: str = Field(default="intfloat/multilingual-e5-small", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")
    chroma_collection: str = Field(default="my_collection", alias="CHROMA_COLLECTION")


    similarity_top_k: int = Field(default=8, alias="SIMILARITY_TOP_K")
    hybrid_enabled: bool = Field(default=True, alias="HYBRID_ENABLED")
    hybrid_alpha: float = Field(default=0.5, alias="HYBRID_ALPHA")
    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    rerank_model: str = Field(default="Xenova/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL")
    rerank_top_n: int = Field(default=4, alias="RERANK_TOP_N")
    adaptive_top_k: bool = Field(default=True, alias="ADAPTIVE_TOP_K")
    query_rewrite_enabled: bool = Field(default=True, alias="QUERY_REWRITE_ENABLED")
    compression_enabled: bool = Field(default=True, alias="COMPRESSION_ENABLED")
    groundedness_enabled: bool = Field(default=True, alias="GROUNDEDNESS_ENABLED")
    groundedness_threshold: float = Field(default=0.35, alias="GROUNDEDNESS_THRESHOLD")

    min_relevance_score: float = Field(default=0.35, ge=0.0, le=1.0, alias="MIN_RELEVANCE_SCORE")

    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")
    cache_max_size: int = Field(default=500, alias="CACHE_MAX_SIZE")
    semantic_cache_enabled: bool = Field(default=True, alias="SEMANTIC_CACHE_ENABLED")
    semantic_cache_threshold: float = Field(default=0.93, alias="SEMANTIC_CACHE_THRESHOLD")

    max_conversation_contexts: int = Field(default=1000, alias="MAX_CONVERSATION_CONTEXTS")
    context_ttl_hours: int = Field(default=24, alias="CONTEXT_TTL_HOURS")
    max_agent_steps: int = Field(default=10, alias="MAX_AGENT_STEPS")

    rate_limit: int = Field(default=50, alias="RATE_LIMIT")
    rate_limit_period: int = Field(default=60, alias="RATE_LIMIT_PERIOD")
    
    chat_rate_limit: int = Field(default=6, gt=0, alias="CHAT_RATE_LIMIT")
    chat_daily_budget: int = Field(default=400, gt=0, alias="CHAT_DAILY_BUDGET")
    max_request_bytes: int = Field(default=16384, gt=0, alias="MAX_REQUEST_BYTES")

    artifacts_dir: str = Field(
        default=str(PROJECT_ROOT / "src" / "artifacts"),
        alias="ARTIFACTS_DIR",
    )


    virtual_today: Optional[str] = Field(default=None, alias="VIRTUAL_TODAY")

    cors_origins_raw: str = Field(
        default="http://localhost:3000,http://localhost:3001",
        alias="CORS_ORIGINS",
    )

    @field_validator("environment")
    @classmethod
    def _normalize_env(cls, v: str) -> str:
        return v.strip().lower()

    @property
    def mongo_url(self) -> str:
        """URL database sesuai environment aktif — dipakai jobs, indexer,
        dan test agar tidak ada yang salah memilih dev/prod secara manual."""
        return self.mongo_url_prod if self.environment == "prod" else self.mongo_url_dev

    
    
@lru_cache
def get_settings() -> Settings:
    return Settings()
