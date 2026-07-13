"""Pydantic BaseSettings cho RAG service production."""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(..., alias="DATABASE_URL")
    backend_internal_base_url: str = Field(..., alias="BACKEND_INTERNAL_BASE_URL")
    internal_api_key: str = Field(..., alias="INTERNAL_API_KEY")

    ollama_base_url: str = Field("http://localhost:11434/v1", alias="OLLAMA_BASE_URL")
    chat_model: str = Field("gemma4b:cloud", alias="CHAT_MODEL")
    embedding_model: str = Field("nomic-embed-text", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(768, alias="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(64, alias="EMBEDDING_BATCH_SIZE")

    chunk_size: int = Field(1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(200, alias="CHUNK_OVERLAP")
    top_k_system: int = Field(5, alias="TOP_K_SYSTEM")
    top_k_hr: int = Field(5, alias="TOP_K_HR")
    request_timeout_seconds: int = Field(120, alias="REQUEST_TIMEOUT_SECONDS")
    debug: bool = Field(False, alias="DEBUG")

    jd_min_chars: int = Field(400, alias="JD_MIN_CHARS")
    jd_max_chars: int = Field(30_000, alias="JD_MAX_CHARS")
    jd_min_words: int = Field(100, alias="JD_MIN_WORDS")
    jd_max_words: int = Field(5_000, alias="JD_MAX_WORDS")


@lru_cache
def get_settings() -> Settings:
    return Settings()
