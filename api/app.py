"""FastAPI app — chỉ internal RAG router + /health."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError

from api import deps
from api.exception_handlers import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from config.settings import get_settings
from models.internal_schemas import HealthResponse
from routers.internal_rag import router as internal_rag_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    deps.startup()
    yield
    deps.shutdown()


app = FastAPI(
    title="IQGS RAG Service",
    version="2.0.0",
    description="Internal RAG API — pgvector ingest, retrieve, generate questions.",
    lifespan=lifespan,
)

app.include_router(internal_rag_router)

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    settings = get_settings()
    config_valid = bool(
        settings.database_url
        and settings.internal_api_key
        and settings.backend_internal_base_url
        and settings.embedding_dimension > 0
    )

    database_status = "down"
    if config_valid:
        try:
            deps.get_vector_store().ping()
            database_status = "up"
        except Exception:
            database_status = "down"

    overall = "ok" if config_valid and database_status == "up" else "degraded"
    return HealthResponse(
        status=overall,
        database=database_status,
        config="valid" if config_valid else "invalid",
    )
