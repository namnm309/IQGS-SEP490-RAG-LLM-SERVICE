"""Singleton wiring: settings, pool, pgvector store, services."""
from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI
from psycopg_pool import ConnectionPool

from config.settings import Settings, get_settings
from services.async_generation_service import AsyncGenerationService
from services.async_ingest_service import AsyncIngestService
from services.backend_callback_client import BackendCallbackClient
from services.chunking_service import ChunkingService
from services.document_downloader import DocumentDownloader
from services.document_parser import DocumentParser
from services.embedding_service import EmbeddingService
from services.plan_generation_service import PlanGenerationService
from services.question_assist_service import QuestionAssistService
from services.evaluate_answer_service import EvaluateAnswerService
from services.question_generation_service import QuestionGenerationService
from services.rag_ingest_service import RagIngestService
from services.jd_parse_service import JdParseService
from services.rag_retrieval_service import RagRetrievalService
from vectorstores.pgvector_store import PgVectorStore

_pool: ConnectionPool | None = None
_vector_store: PgVectorStore | None = None
_openai_client: OpenAI | None = None
_ingest_service: RagIngestService | None = None
_question_service: QuestionGenerationService | None = None
_plan_service: PlanGenerationService | None = None
_jd_parse_service: JdParseService | None = None
_async_generation_service: AsyncGenerationService | None = None
_async_ingest_service: AsyncIngestService | None = None
_question_assist_service: QuestionAssistService | None = None
_evaluate_answer_service: EvaluateAnswerService | None = None


@dataclass
class AppServices:
    settings: Settings
    pool: ConnectionPool
    vector_store: PgVectorStore
    ingest_service: RagIngestService
    question_service: QuestionGenerationService
    plan_service: PlanGenerationService


def startup() -> None:
    global _pool, _vector_store, _openai_client, _ingest_service, _question_service, _plan_service, _jd_parse_service, _async_generation_service, _async_ingest_service, _question_assist_service, _evaluate_answer_service

    settings = get_settings()
    _pool = ConnectionPool(conninfo=settings.database_url, min_size=1, max_size=10, open=True)
    _vector_store = PgVectorStore(_pool, settings)
    _openai_client = OpenAI(api_key="ollama", base_url=settings.ollama_base_url)

    embedding_service = EmbeddingService(_openai_client, settings)
    callback_client = BackendCallbackClient(settings)
    downloader = DocumentDownloader(settings)
    parser = DocumentParser()
    chunking = ChunkingService(settings)
    retrieval = RagRetrievalService(_vector_store, embedding_service, settings)

    _ingest_service = RagIngestService(
        vector_store=_vector_store,
        downloader=downloader,
        parser=parser,
        chunking=chunking,
        embedding=embedding_service,
        callback_client=callback_client,
    )
    _question_service = QuestionGenerationService(
        retrieval=retrieval,
        client=_openai_client,
        settings=settings,
    )
    _plan_service = PlanGenerationService(
        retrieval=retrieval,
        client=_openai_client,
        settings=settings,
    )
    _jd_parse_service = JdParseService(parser=parser, settings=settings)
    _async_generation_service = AsyncGenerationService(
        plan_service=_plan_service,
        question_service=_question_service,
        callback_client=callback_client,
    )
    _async_ingest_service = AsyncIngestService(ingest_service=_ingest_service)
    _question_assist_service = QuestionAssistService(client=_openai_client, settings=settings)
    _evaluate_answer_service = EvaluateAnswerService(client=_openai_client, settings=settings)


def shutdown() -> None:
    global _pool, _vector_store, _openai_client, _ingest_service, _question_service, _plan_service, _jd_parse_service, _async_generation_service, _async_ingest_service, _question_assist_service, _evaluate_answer_service

    if _pool is not None:
        _pool.close()
    _pool = None
    _vector_store = None
    _openai_client = None
    _ingest_service = None
    _question_service = None
    _plan_service = None
    _jd_parse_service = None
    _async_generation_service = None
    _async_ingest_service = None
    _question_assist_service = None
    _evaluate_answer_service = None


def get_services() -> AppServices:
    if (
        _pool is None
        or _vector_store is None
        or _ingest_service is None
        or _question_service is None
        or _plan_service is None
    ):
        raise RuntimeError("Application services chưa được khởi tạo")
    return AppServices(
        settings=get_settings(),
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
    return _ingest_service


def get_question_service() -> QuestionGenerationService:
    if _question_service is None:
        raise RuntimeError("Question service chưa được khởi tạo")
    return _question_service


def get_plan_service() -> PlanGenerationService:
    if _plan_service is None:
        raise RuntimeError("Plan service chưa được khởi tạo")
    return _plan_service


def get_jd_parse_service() -> JdParseService:
    if _jd_parse_service is None:
        raise RuntimeError("JD parse service chưa được khởi tạo")
    return _jd_parse_service


def get_async_generation_service() -> AsyncGenerationService:
    if _async_generation_service is None:
        raise RuntimeError("Async generation service chưa được khởi tạo")
    return _async_generation_service


def get_async_ingest_service() -> AsyncIngestService:
    if _async_ingest_service is None:
        raise RuntimeError("Async ingest service chưa được khởi tạo")
    return _async_ingest_service


def get_question_assist_service() -> QuestionAssistService:
    if _question_assist_service is None:
        raise RuntimeError("Question assist service chưa được khởi tạo")
    return _question_assist_service


def get_evaluate_answer_service() -> EvaluateAnswerService:
    if _evaluate_answer_service is None:
        raise RuntimeError("Evaluate answer service chưa được khởi tạo")
    return _evaluate_answer_service
