"""Xử lý ingest nền và callback kết quả về Backend."""
from __future__ import annotations

import logging

from models.internal_schemas import IngestRequest
from services.rag_ingest_service import RagIngestService

logger = logging.getLogger(__name__)


class AsyncIngestService:
    def __init__(self, ingest_service: RagIngestService):
        self._ingest = ingest_service

    def run_ingest(self, request: IngestRequest) -> None:
        """Chạy pipeline ingest; lỗi đã được callback về Backend trong execute_ingest."""
        try:
            self._ingest.execute_ingest(request)
        except Exception:
            logger.exception(
                "Async ingest thất bại không mong đợi cho document %s",
                request.document_id,
            )
