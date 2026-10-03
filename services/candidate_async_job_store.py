"""Kho job nền (in-memory) cho sinh đề Candidate/Coach.

Vì sao có file này: Cloudflare Tunnel cắt mọi request chờ > ~100s (lỗi 524), mà sinh đề Coach bằng LLM
cloud có thể mất 2-4 phút. Nên RAG nhận job rồi trả 202 ngay, chạy nền, còn BE hỏi trạng thái
(GET) mỗi vài giây — mỗi request đều ngắn nên không bị Cloudflare cắt.

Lưu in-memory vì RAG chạy 1 process uvicorn và BE chỉ cần kết quả trong vài phút. Nếu RAG restart giữa
chừng thì job mất → BE nhận 404 và báo lỗi để user bấm thử lại (giống hành vi cũ khi RAG chết).
"""
from __future__ import annotations

import threading
import time
from typing import Any

# Giữ kết quả 30 phút là đủ cho BE lấy; quá hạn thì dọn để không phình RAM.
_TTL_SECONDS = 30 * 60

STATUS_PROCESSING = "PROCESSING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"


class CandidateAsyncJobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def start(self, job_id: str) -> None:
        with self._lock:
            self._purge_expired_locked()
            self._jobs[job_id] = {
                "status": STATUS_PROCESSING,
                "result": None,
                "updated_at": time.time(),
            }

    def finish(self, job_id: str, *, success: bool, result: dict[str, Any]) -> None:
        with self._lock:
            self._jobs[job_id] = {
                "status": STATUS_COMPLETED if success else STATUS_FAILED,
                "result": result,
                "updated_at": time.time(),
            }

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            self._purge_expired_locked()
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def _purge_expired_locked(self) -> None:
        now = time.time()
        expired = [k for k, v in self._jobs.items() if now - v["updated_at"] > _TTL_SECONDS]
        for key in expired:
            del self._jobs[key]


# Một kho dùng chung cho cả process.
candidate_async_job_store = CandidateAsyncJobStore()
