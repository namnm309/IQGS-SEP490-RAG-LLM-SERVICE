"""Trả lỗi JSON có cấu trúc — BE đọc detail/stage thay vì chỉ HTTP status."""
from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def _normalize_detail(detail: object) -> dict:
    if isinstance(detail, dict):
        errors = detail.get("errors")
        if not isinstance(errors, list):
            errors = []
        return {
            "error": str(detail.get("error") or detail.get("detail") or "RAG error"),
            "detail": str(detail.get("detail") or detail.get("error") or ""),
            "stage": detail.get("stage"),
            "exceptionType": detail.get("exceptionType"),
            "errors": [str(e) for e in errors],
        }
    text = str(detail)
    return {
        "error": text,
        "detail": text,
        "stage": None,
        "exceptionType": "HTTPException",
        "errors": [text],
    }


async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    body = _normalize_detail(exc.detail)
    return JSONResponse(status_code=exc.status_code, content=body)


async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    messages = "; ".join(
        f"{'.'.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg')}"
        for err in exc.errors()
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "detail": messages,
            "stage": "VALIDATION",
            "exceptionType": "RequestValidationError",
            "errors": [messages],
        },
    )


async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal RAG error",
            "detail": str(exc),
            "stage": "UNHANDLED",
            "exceptionType": type(exc).__name__,
            "errors": [str(exc)],
        },
    )
