"""Unit tests cho runtime settings merge — SCRUM-379."""
from __future__ import annotations

from unittest.mock import MagicMock

from config.runtime_settings import RuntimeRow, apply_runtime_overrides, clear_runtime_cache
from config.settings import Settings


def _settings() -> Settings:
    return Settings(
        DATABASE_URL="postgresql://postgres:postgres@localhost:5432/iqgs",
        BACKEND_INTERNAL_BASE_URL="http://localhost:5000",
        INTERNAL_API_KEY="test-secret",
        EMBEDDING_DIMENSION=768,
        CHAT_MODEL="gemma4b:cloud",
        OLLAMA_BASE_URL="http://localhost:11434/v1",
        CHAT_BASE_URL="http://localhost:11434/v1",
        TEMPERATURE=0.3,
        TOP_K_SYSTEM=5,
        TOP_K_HR=5,
    )


def test_apply_openrouter_chat_keeps_local_embed(monkeypatch) -> None:
    clear_runtime_cache()
    settings = _settings()
    pool = MagicMock()

    row = RuntimeRow(
        llm_provider="openrouter",
        chat_base_url="https://openrouter.ai/api/v1",
        chat_api_key="sk-or-test",
        ollama_base_url="http://localhost:11434/v1",
        ollama_api_key="ollama",
        chat_model="openai/gpt-4o-mini",
        temperature=0.5,
        top_k_system=8,
        top_k_hr=3,
        request_timeout_seconds=90,
    )

    monkeypatch.setattr(
        "config.runtime_settings.get_runtime_row",
        lambda _pool, force=False: row,
    )

    applied, chat_changed, embed_changed = apply_runtime_overrides(settings, pool, force=True)
    assert applied is True
    assert chat_changed is True
    assert settings.llm_provider == "openrouter"
    assert settings.chat_base_url == "https://openrouter.ai/api/v1"
    assert settings.chat_model == "openai/gpt-4o-mini"
    assert settings.ollama_base_url == "http://localhost:11434/v1"


def test_apply_runtime_overrides_none_row(monkeypatch) -> None:
    clear_runtime_cache()
    settings = _settings()
    pool = MagicMock()
    monkeypatch.setattr(
        "config.runtime_settings.get_runtime_row",
        lambda _pool, force=False: None,
    )
    applied, chat_changed, embed_changed = apply_runtime_overrides(settings, pool, force=True)
    assert applied is False
    assert chat_changed is False
    assert embed_changed is False
