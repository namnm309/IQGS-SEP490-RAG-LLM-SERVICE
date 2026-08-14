"""Singleton wiring: settings, pool, pgvector store, services.

SCRUM-378: runtime overrides từ DB + hot-reload Ollama client khi URL/key đổi.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI
from psycopg_pool import ConnectionPool

from config.runtime_settings import apply_runtime_overrides, clear_runtime_cache
from config.settings import Settings, get_settings
from services.async_generation_service import AsyncGenerationService
from services.async_ingest_service import AsyncIngestService
from services.backend_callback_client import BackendCallbackClient
from services.chunking_service import ChunkingService
from services.cv_parse_service import CvParseService
from services.document_downloader import DocumentDownloader
from services.document_parser import DocumentParser
from services.embedding_service import EmbeddingService
from services.plan_generation_service import PlanGenerationService
from services.question_assist_service import QuestionAssistService
from services.evaluate_answer_service import EvaluateAnswerService
from services.practice_session_insight_service import PracticeSessionInsightService
from services.question_generation_service import QuestionGenerationService
from services.rag_ingest_service import RagIngestService
from services.jd_parse_service import JdParseService
from services.rag_retrieval_service import RagRetrievalService
from vectorstores.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
_vector_store: PgVectorStore | None = None
_chat_client: OpenAI | None = None
_embed_client: OpenAI | None = None
# Alias tương thích code cũ / tests
_openai_client: OpenAI | None = None
_ingest_service: RagIngestService | None = None
_question_service: QuestionGenerationService | None = None
_plan_service: PlanGenerationService | None = None
_jd_parse_service: JdParseService | None = None
_cv_parse_service: CvParseService | None = None
_async_generation_service: AsyncGenerationService | None = None
_async_ingest_service: AsyncIngestService | None = None
_question_assist_service: QuestionAssistService | None = None
_evaluate_answer_service: EvaluateAnswerService | None = None
_practice_session_insight_service: PracticeSessionInsightService | None = None
_retrieval_service: RagRetrievalService | None = None
_embedding_service: EmbeddingService | None = None
_settings_ref: Settings | None = None


@dataclass
class AppServices:
    settings: Settings
    pool: ConnectionPool
    vector_store: PgVectorStore
    ingest_service: RagIngestService
    question_service: QuestionGenerationService
    plan_service: PlanGenerationService


def _create_chat_client(settings: Settings) -> OpenAI:
    """Client cho LLM chat — Ollama hoặc OpenRouter (SCRUM-379)."""
    kwargs: dict = {
        "api_key": settings.chat_api_key or "ollama",
        "base_url": settings.chat_base_url or settings.ollama_base_url,
        "timeout": settings.request_timeout_seconds,
    }
    if "openrouter.ai" in (settings.chat_base_url or "").lower():
        kwargs["default_headers"] = {
            "HTTP-Referer": "https://iqgs.local",
            "X-Title": "IQGS RAG",
        }
    return OpenAI(**kwargs)


def _create_embed_client(settings: Settings) -> OpenAI:
    """Client embedding — luôn Ollama local."""
    return OpenAI(
        api_key=settings.ollama_api_key or "ollama",
        base_url=settings.ollama_base_url,
        timeout=settings.request_timeout_seconds,
    )


def _wire_chat_client(client: OpenAI) -> None:
    if _question_service is not None:
        _question_service._client = client  # noqa: SLF001
    if _plan_service is not None:
        _plan_service._client = client  # noqa: SLF001
    if _cv_parse_service is not None:
        _cv_parse_service._client = client  # noqa: SLF001
    if _question_assist_service is not None:
        _question_assist_service._client = client  # noqa: SLF001
    if _evaluate_answer_service is not None:
        _evaluate_answer_service._client = client  # noqa: SLF001
    if _practice_session_insight_service is not None:
        _practice_session_insight_service._client = client  # noqa: SLF001


def _wire_embed_client(client: OpenAI) -> None:
    if _embedding_service is not None:
        _embedding_service._client = client  # noqa: SLF001


def refresh_runtime_config(*, force: bool = False) -> dict:
    """Đọc DB overrides; recreate chat và/hoặc embed client khi URL/key đổi."""
    global _chat_client, _embed_client, _openai_client

    if _pool is None or _settings_ref is None:
        raise RuntimeError("Application services chưa được khởi tạo")

    applied, chat_changed, embed_changed = apply_runtime_overrides(
        _settings_ref, _pool, force=force
    )

    if chat_changed or _chat_client is None:
        logger.info(
            "Tạo lại chat client (provider=%s url=%s)",
            _settings_ref.llm_provider,
            _settings_ref.chat_base_url,
        )
        _chat_client = _create_chat_client(_settings_ref)
        _openai_client = _chat_client
        _wire_chat_client(_chat_client)

    if embed_changed or _embed_client is None:
        logger.info("Tạo lại embed client (url=%s)", _settings_ref.ollama_base_url)
        _embed_client = _create_embed_client(_settings_ref)
        _wire_embed_client(_embed_client)

    return {
        "applied": applied,
        "connectionChanged": chat_changed or embed_changed,
        "chatModel": _settings_ref.chat_model,
        "llmProvider": _settings_ref.llm_provider,
        "chatBaseUrl": _settings_ref.chat_base_url,
        "ollamaBaseUrl": _settings_ref.ollama_base_url,
        "temperature": _settings_ref.temperature,
        "topKSystem": _settings_ref.top_k_system,
        "topKHr": _settings_ref.top_k_hr,
    }


def reload_runtime_config() -> dict:
    """Force reload từ DB (sau Admin PUT)."""
    clear_runtime_cache()
    return refresh_runtime_config(force=True)


def startup() -> None:
    global _pool, _vector_store, _chat_client, _embed_client, _openai_client
    global _ingest_service, _question_service, _plan_service
    global _jd_parse_service, _cv_parse_service, _async_generation_service, _async_ingest_service
    global _question_assist_service, _evaluate_answer_service, _practice_session_insight_service
    global _retrieval_service, _embedding_service, _settings_ref

    settings = get_settings()
    # Đồng bộ mặc định chat ← ollama nếu chưa set env riêng
    if not getattr(settings, "chat_base_url", None):
        settings.chat_base_url = settings.ollama_base_url
    if not getattr(settings, "chat_api_key", None) or settings.chat_api_key == "ollama":
        if settings.llm_provider == "ollama":
            settings.chat_api_key = settings.ollama_api_key or "ollama"

    _settings_ref = settings
    _pool = ConnectionPool(conninfo=settings.database_url, min_size=1, max_size=10, open=True)

    apply_runtime_overrides(settings, _pool, force=True)

    _vector_store = PgVectorStore(_pool, settings)
    _chat_client = _create_chat_client(settings)
    _embed_client = _create_embed_client(settings)
    _openai_client = _chat_client

    _embedding_service = EmbeddingService(_embed_client, settings)
    callback_client = BackendCallbackClient(settings)
    downloader = DocumentDownloader(settings)
    parser = DocumentParser()
    chunking = ChunkingService(settings)
    _retrieval_service = RagRetrievalService(_vector_store, _embedding_service, settings)

    _ingest_service = RagIngestService(
        vector_store=_vector_store,
        downloader=downloader,
        parser=parser,
        chunking=chunking,
        embedding=_embedding_service,
        callback_client=callback_client,
    )
    _question_service = QuestionGenerationService(
        retrieval=_retrieval_service,
        client=_chat_client,
        settings=settings,
    )
    _plan_service = PlanGenerationService(
        retrieval=_retrieval_service,
        client=_chat_client,
        settings=settings,
    )
    _jd_parse_service = JdParseService(parser=parser, settings=settings)
    _cv_parse_service = CvParseService(parser=parser, client=_chat_client, settings=settings)
    _async_generation_service = AsyncGenerationService(
        plan_service=_plan_service,
        question_service=_question_service,
        callback_client=callback_client,
    )
    _async_ingest_service = AsyncIngestService(ingest_service=_ingest_service)
    _question_assist_service = QuestionAssistService(client=_chat_client, settings=settings)
    _evaluate_answer_service = EvaluateAnswerService(client=_chat_client, settings=settings)
    _practice_session_insight_service = PracticeSessionInsightService(
        client=_chat_client, settings=settings
    )


def shutdown() -> None:
    global _pool, _vector_store, _chat_client, _embed_client, _openai_client
    global _ingest_service, _question_service, _plan_service
    global _jd_parse_service, _cv_parse_service, _async_generation_service, _async_ingest_service
    global _question_assist_service, _evaluate_answer_service, _practice_session_insight_service
    global _retrieval_service, _embedding_service, _settings_ref

    if _pool is not None:
        _pool.close()
    _pool = None
    _vector_store = None
    _chat_client = None
    _embed_client = None
    _openai_client = None
    _ingest_service = None
    _question_service = None
    _plan_service = None
    _jd_parse_service = None
    _cv_parse_service = None
    _async_generation_service = None
    _async_ingest_service = None
    _question_assist_service = None
    _evaluate_answer_service = None
    _practice_session_insight_service = None
    _retrieval_service = None
    _embedding_service = None
    _settings_ref = None
    clear_runtime_cache()


def get_services() -> AppServices:
    if (
        _pool is None
        or _vector_store is None
        or _ingest_service is None
        or _question_service is None
        or _plan_service is None
        or _settings_ref is None
    ):
        raise RuntimeError("Application services chưa được khởi tạo")
    # Refresh TTL trước khi trả settings (không force)
    try:
        refresh_runtime_config(force=False)
    except Exception:
        logger.exception("refresh_runtime_config thất bại")
    return AppServices(
        settings=_settings_ref,
        pool=_pool,
        vector_store=_vector_store,
        ingest_service=_ingest_service,
        question_service=_question_service,
        plan_service=_plan_service,
    )


def get_vector_store() -> PgVectorStore:
    if _vector_store is None:
        raise RuntimeError("Vector store chưa được khởi tạo")
    return _vector_store


def get_ingest_service() -> RagIngestService:
    if _ingest_service is None:
        raise RuntimeError("Ingest service chưa được khởi tạo")
    try:
        refresh_runtime_config(force=False)
    except Exception:
        pass
    return _ingest_service


def get_question_service() -> QuestionGenerationService:
    if _question_service is None:
        raise RuntimeError("Question service chưa được khởi tạo")
    try:
        refresh_runtime_config(force=False)
    except Exception:
        pass
    return _question_service


def get_plan_service() -> PlanGenerationService:
    if _plan_service is None:
        raise RuntimeError("Plan service chưa được khởi tạo")
    try:
        refresh_runtime_config(force=False)
    except Exception:
        pass
    return _plan_service


def get_jd_parse_service() -> JdParseService:
    if _jd_parse_service is None:
        raise RuntimeError("JD parse service chưa được khởi tạo")
    return _jd_parse_service


def get_cv_parse_service() -> CvParseService:
    if _cv_parse_service is None:
        raise RuntimeError("CV parse service chưa được khởi tạo")
    try:
        refresh_runtime_config(force=False)
    except Exception:
        pass
    return _cv_parse_service


def get_async_generation_service() -> AsyncGenerationService:
    if _async_generation_service is None:
        raise RuntimeError("Async generation service chưa được khởi tạo")
    try:
        refresh_runtime_config(force=False)
    except Exception:
        pass
    return _async_generation_service


def get_async_ingest_service() -> AsyncIngestService:
    if _async_ingest_service is None:
        raise RuntimeError("Async ingest service chưa được khởi tạo")
    return _async_ingest_service


def get_question_assist_service() -> QuestionAssistService:
    if _question_assist_service is None:
        raise RuntimeError("Question assist service chưa được khởi tạo")
    try:
        refresh_runtime_config(force=False)
    except Exception:
        pass
    return _question_assist_service


def get_evaluate_answer_service() -> EvaluateAnswerService:
    if _evaluate_answer_service is None:
        raise RuntimeError("Evaluate answer service chưa được khởi tạo")
    try:
        refresh_runtime_config(force=False)
    except Exception:
        pass
    return _evaluate_answer_service


def get_practice_session_insight_service() -> PracticeSessionInsightService:
    if _practice_session_insight_service is None:
        raise RuntimeError("Practice session insight service chưa được khởi tạo")
    try:
        refresh_runtime_config(force=False)
    except Exception:
        pass
    return _practice_session_insight_service


def get_openai_client() -> OpenAI:
    if _openai_client is None:
        raise RuntimeError("OpenAI client chưa được khởi tạo")
    return _openai_client


def get_settings_ref() -> Settings:
    if _settings_ref is None:
        raise RuntimeError("Settings chưa được khởi tạo")
    return _settings_ref
