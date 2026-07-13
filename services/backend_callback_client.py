"""Callback PATCH status tới Backend — lỗi chỉ log warning, không raise."""
from __future__ import annotations

import logging
from typing import Literal

import httpx

from config.settings import Settings

logger = logging.getLogger(__name__)

DocumentStatus = Literal["PROCESSING", "COMPLETED", "FAILED"]


class BackendCallbackClient:
    def __init__(self, settings: Settings):
        self._settings = settings

    def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        *,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        url = (
            f"{self._settings.backend_internal_base_url.rstrip('/')}"
            f"/internal/knowledge-documents/{document_id}/status"
        )
        payload: dict[str, object] = {"status": status}
        if chunk_count is not None:
            payload["chunkCount"] = chunk_count
        if error_message:
            payload["errorMessage"] = error_message

        headers = {"X-Internal-Api-Key": self._settings.internal_api_key}
        timeout = httpx.Timeout(self._settings.request_timeout_seconds)

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.patch(url, json=payload, headers=headers)
                if response.status_code >= 400:
                    logger.warning(
                        "Callback %s thất bại HTTP %s: %s",
                        status,
                        response.status_code,
                        response.text[:500],
                    )
        except httpx.HTTPError as exc:
            logger.warning("Callback %s lỗi mạng cho document %s: %s", status, document_id, exc)

    def notify_generation_result(self, job_id: str, payload: dict[str, object]) -> None:
        url = (
            f"{self._settings.backend_internal_base_url.rstrip('/')}"
            f"/internal/question-generation-jobs/{job_id}/generation-result"
        )
        headers = {"X-Internal-Api-Key": self._settings.internal_api_key}
        timeout = httpx.Timeout(self._settings.request_timeout_seconds)

        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.patch(url, json=payload, headers=headers)
                if response.status_code >= 400:
                    logger.warning(
                        "Callback generation-result thất bại HTTP %s cho job %s: %s",
                        response.status_code,
                        job_id,
                        response.text[:500],
                    )
        except httpx.HTTPError as exc:
            logger.warning(
                "Callback generation-result lỗi mạng cho job %s: %s", job_id, exc
            )
