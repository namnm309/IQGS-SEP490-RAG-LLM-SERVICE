"""Load RAG runtime overrides từ bảng rag_runtime_settings (shared DB với BE).

SCRUM-379: Chat LLM (Ollama|OpenRouter) tách khỏi Embedding (luôn Ollama local).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool

    from config.settings import Settings

logger = logging.getLogger(__name__)

_TTL_SECONDS = 30.0

_cache_loaded_at: float = 0.0
_cache_row: "RuntimeRow | None" = None


@dataclass(frozen=True)
class RuntimeRow:
    llm_provider: str | None
    chat_base_url: str | None
    chat_api_key: str | None
    ollama_base_url: str | None
    ollama_api_key: str | None
    chat_model: str | None
    temperature: float | None
    top_k_system: int | None
    top_k_hr: int | None
    request_timeout_seconds: int | None


def _fetch_row(pool: ConnectionPool) -> RuntimeRow | None:
    """SELECT singleton row; trả None nếu bảng/row chưa tồn tại."""
    sql = """
        SELECT llm_provider, chat_base_url, chat_api_key,
               ollama_base_url, ollama_api_key, chat_model, temperature,
               top_k_system, top_k_hr, request_timeout_seconds
        FROM rag_runtime_settings
        ORDER BY id
        LIMIT 1
    """
    # Fallback cột cũ (trước migration SCRUM-379)
    sql_legacy = """
        SELECT ollama_base_url, ollama_api_key, chat_model, temperature,
               top_k_system, top_k_hr, request_timeout_seconds
        FROM rag_runtime_settings
        ORDER BY id
        LIMIT 1
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(sql)
                    row = cur.fetchone()
                    if not row:
                        return None
                    return RuntimeRow(
                        llm_provider=row[0],
                        chat_base_url=row[1],
                        chat_api_key=row[2],
                        ollama_base_url=row[3],
                        ollama_api_key=row[4],
                        chat_model=row[5],
                        temperature=float(row[6]) if row[6] is not None else None,
                        top_k_system=int(row[7]) if row[7] is not None else None,
                        top_k_hr=int(row[8]) if row[8] is not None else None,
                        request_timeout_seconds=int(row[9]) if row[9] is not None else None,
                    )
                except Exception:
                    conn.rollback()
                    cur.execute(sql_legacy)
                    row = cur.fetchone()
                    if not row:
                        return None
                    # Legacy: 1 URL cho cả chat+embed
                    return RuntimeRow(
                        llm_provider="openrouter" if row[0] and "openrouter.ai" in str(row[0]).lower() else "ollama",
                        chat_base_url=row[0],
                        chat_api_key=row[1],
                        ollama_base_url="http://localhost:11434/v1"
                        if row[0] and "openrouter.ai" in str(row[0]).lower()
                        else row[0],
                        ollama_api_key=None if row[0] and "openrouter.ai" in str(row[0]).lower() else row[1],
                        chat_model=row[2],
                        temperature=float(row[3]) if row[3] is not None else None,
                        top_k_system=int(row[4]) if row[4] is not None else None,
                        top_k_hr=int(row[5]) if row[5] is not None else None,
                        request_timeout_seconds=int(row[6]) if row[6] is not None else None,
                    )
    except Exception as exc:
        logger.warning("Không đọc được rag_runtime_settings, dùng env: %s", exc)
        return None


def get_runtime_row(pool: ConnectionPool, *, force: bool = False) -> RuntimeRow | None:
    global _cache_loaded_at, _cache_row
    now = time.monotonic()
    if not force and _cache_row is not None and (now - _cache_loaded_at) < _TTL_SECONDS:
        return _cache_row
    _cache_row = _fetch_row(pool)
    _cache_loaded_at = now
    return _cache_row


def apply_runtime_overrides(settings: Settings, pool: ConnectionPool, *, force: bool = False) -> tuple[bool, bool, bool]:
    """Áp override từ DB lên Settings.

    Returns:
        (applied, chat_connection_changed, embed_connection_changed)
    """
    row = get_runtime_row(pool, force=force)
    if row is None:
        return False, False, False

    old_chat_url = getattr(settings, "chat_base_url", settings.ollama_base_url)
    old_chat_key = getattr(settings, "chat_api_key", settings.ollama_api_key) or "ollama"
    old_embed_url = settings.ollama_base_url
    old_embed_key = settings.ollama_api_key or "ollama"

    if row.llm_provider:
        settings.llm_provider = row.llm_provider.strip().lower() or "ollama"

    if row.chat_base_url:
        settings.chat_base_url = row.chat_base_url.rstrip("/")
    elif row.ollama_base_url and settings.llm_provider == "ollama":
        settings.chat_base_url = row.ollama_base_url.rstrip("/")

    if row.chat_api_key is not None:
        settings.chat_api_key = row.chat_api_key.strip() or "ollama"
    elif row.ollama_api_key is not None and settings.llm_provider == "ollama":
        settings.chat_api_key = row.ollama_api_key.strip() or "ollama"

    # Embedding: luôn Ollama — không bao giờ lấy OpenRouter URL
    if row.ollama_base_url and "openrouter.ai" not in row.ollama_base_url.lower():
        settings.ollama_base_url = row.ollama_base_url.rstrip("/")
    elif "openrouter.ai" in settings.ollama_base_url.lower():
        settings.ollama_base_url = "http://localhost:11434/v1"

    if row.ollama_api_key is not None:
        settings.ollama_api_key = row.ollama_api_key.strip() or "ollama"

    if row.chat_model:
        settings.chat_model = row.chat_model
    if row.temperature is not None:
        settings.temperature = row.temperature
    if row.top_k_system is not None:
        settings.top_k_system = row.top_k_system
    if row.top_k_hr is not None:
        settings.top_k_hr = row.top_k_hr
    if row.request_timeout_seconds is not None:
        settings.request_timeout_seconds = row.request_timeout_seconds

    new_chat_key = settings.chat_api_key or "ollama"
    new_embed_key = settings.ollama_api_key or "ollama"
    chat_changed = settings.chat_base_url != old_chat_url or new_chat_key != old_chat_key
    embed_changed = settings.ollama_base_url != old_embed_url or new_embed_key != old_embed_key
    return True, chat_changed, embed_changed


def clear_runtime_cache() -> None:
    global _cache_loaded_at, _cache_row
    _cache_loaded_at = 0.0
    _cache_row = None
