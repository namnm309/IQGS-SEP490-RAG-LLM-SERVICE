"""Xử lý sinh plan/câu hỏi nền và callback kết quả về Backend."""
from __future__ import annotations

import logging
from typing import Any

from models.internal_schemas import (
    GeneratePlanRequest,
    GenerateQuestionsFromPlanRequest,
)
from services.backend_callback_client import BackendCallbackClient
from services.plan_generation_service import PlanGenerationService
from services.question_generation_service import QuestionGenerationService

logger = logging.getLogger(__name__)

PHASE_PLAN = "PLAN"
PHASE_QUESTIONS = "QUESTIONS"


class AsyncGenerationService:
    def __init__(
        self,
        plan_service: PlanGenerationService,
        question_service: QuestionGenerationService,
        callback_client: BackendCallbackClient,
    ):
        self._plan = plan_service
        self._questions = question_service
        self._callback = callback_client

    def run_plan(self, job_id: str, request: GeneratePlanRequest) -> None:
        try:
            result = self._plan.generate(request)
            if result.success and result.plan is not None:
                payload: dict[str, Any] = {
                    "phase": PHASE_PLAN,
                    "success": True,
                    "plan": result.plan.model_dump(by_alias=True),
                    "processingTimeMs": result.processing_time_ms,
                }
            else:
                payload = {
                    "phase": PHASE_PLAN,
                    "success": False,
                    "error": result.error or "Lỗi sinh plan",
                    "detail": result.detail or result.error,
                    "stage": result.stage or "PLAN_GENERATION",
                    "source": "RAG",
                    "errors": result.errors or ([result.error] if result.error else []),
                    "processingTimeMs": result.processing_time_ms,
                }
            self._callback.notify_generation_result(job_id, payload)
        except Exception as exc:
            logger.exception("Async plan generation failed for job %s", job_id)
            self._callback.notify_generation_result(
                job_id,
                {
                    "phase": PHASE_PLAN,
                    "success": False,
                    "error": "Lỗi sinh plan",
                    "detail": str(exc),
                    "stage": "PLAN_GENERATION",
                    "source": "RAG",
                    "errors": [str(exc)],
                },
            )

    def run_questions_from_plan(
        self, job_id: str, request: GenerateQuestionsFromPlanRequest
    ) -> None:
        try:
            result = self._questions.generate_from_plan(request)
            if result.success and result.questions:
                payload: dict[str, Any] = {
                    "phase": PHASE_QUESTIONS,
                    "success": True,
                    "questions": [
                        q.model_dump(by_alias=True) for q in result.questions
                    ],
                    "processingTimeMs": result.processing_time_ms,
                }
            else:
                payload = {
                    "phase": PHASE_QUESTIONS,
                    "success": False,
                    "error": result.error or "Lỗi sinh câu hỏi",
                    "detail": result.error,
                    "stage": "QUESTION_GENERATION",
                    "source": "RAG",
                    "errors": [result.error] if result.error else ["RAG generate-questions-from-plan thất bại."],
                    "processingTimeMs": result.processing_time_ms,
                }
            self._callback.notify_generation_result(job_id, payload)
        except Exception as exc:
            logger.exception("Async question generation failed for job %s", job_id)
            self._callback.notify_generation_result(
                job_id,
                {
                    "phase": PHASE_QUESTIONS,
                    "success": False,
                    "error": "Lỗi sinh câu hỏi",
                    "detail": str(exc),
                    "stage": "QUESTION_GENERATION",
                    "source": "RAG",
                    "errors": [str(exc)],
                },
            )
