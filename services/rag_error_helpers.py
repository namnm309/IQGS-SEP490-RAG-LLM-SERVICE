"""Helper tạo payload lỗi structured cho RAG internal API."""
from __future__ import annotations


def build_rag_error_detail(
    *,
    error: str,
    detail: str | None = None,
    stage: str | None = None,
    exception_type: str | None = None,
    errors: list[str] | None = None,
) -> dict:
    """Dict dùng làm HTTPException.detail — BE deserialize qua RagErrorResponse."""
    err_list = list(errors) if errors else []
    if not err_list and detail:
        err_list = [detail]
    return {
        "error": error,
        "detail": detail or (err_list[0] if err_list else error),
        "stage": stage,
        "exceptionType": exception_type,
        "errors": err_list,
    }
