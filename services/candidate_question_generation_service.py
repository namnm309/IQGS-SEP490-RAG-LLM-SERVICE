"""Sinh câu hỏi luyện tập Candidate — SYSTEM-only, prompt coach / jd_practice."""
from __future__ import annotations

import logging
import time

from models.internal_schemas import (
    GenerateQuestionsFromPlanRequest,
    GenerateQuestionsFromPlanResponse,
)
from services.candidate_generation_prompts import candidate_questions_system_prompt
from services.json_output_parser import is_retryable_json_error
from services.question_generation_service import (
    QuestionGenerationService,
    _validate_approved_plan,
    _validate_owner_and_jd,
)
from services.rag_context_helpers import build_chunk_lookup, format_retrieved_context
from helpers.language_prompt import language_instruction_block
from vectorstores.base import RetrievedChunk

logger = logging.getLogger(__name__)


def _effective_jd(request: GenerateQuestionsFromPlanRequest) -> str:
    cv = getattr(request, "cv_context", None)
    if cv and str(cv).strip():
        return str(cv).strip()
    return (request.job_description or "").strip()


def _effective_note(request: GenerateQuestionsFromPlanRequest) -> str | None:
    note = getattr(request, "candidate_note", None)
    if note and str(note).strip():
        return str(note).strip()
    if request.hr_note and request.hr_note.strip():
        return request.hr_note.strip()
    return None


class CandidateQuestionGenerationService(QuestionGenerationService):
    def generate_from_plan(
        self, request: GenerateQuestionsFromPlanRequest
    ) -> GenerateQuestionsFromPlanResponse:
        start = time.time()
        jd = _effective_jd(request)
        if not (request.job_description or "").strip() and jd:
            request.job_description = jd

        validation_error = _validate_owner_and_jd(request.owner_id, request.job_description)
        if validation_error:
            return GenerateQuestionsFromPlanResponse(
                success=False,
                error=validation_error,
                processing_time_ms=(time.time() - start) * 1000,
            )

        plan_error = _validate_approved_plan(request.approved_plan)
        if plan_error:
            return GenerateQuestionsFromPlanResponse(
                success=False,
                error=plan_error,
                processing_time_ms=(time.time() - start) * 1000,
            )

        note = _effective_note(request)
        if note:
            request.hr_note = note

        system_chunks: list[RetrievedChunk] = []
        try:
            retrieve_sys = getattr(self._retrieval, "retrieve_system_only", None)
            if retrieve_sys is not None:
                system_chunks = retrieve_sys(jd, query_extra=note) or []
            else:
                system_chunks, _ = self._retrieval.retrieve_for_job(
                    jd, request.owner_id.strip()
                )
                system_chunks = system_chunks or []
        except Exception as exc:
            logger.warning(
                "Candidate questions SYSTEM retrieve thất bại — vẫn sinh câu: %s", exc
            )
            system_chunks = []

        hr_chunks: list[RetrievedChunk] = []
        all_chunks = list(system_chunks)

        from services.question_generation_service import _BATCH_THRESHOLD

        total = request.approved_plan.total_questions
        if total > _BATCH_THRESHOLD:
            questions, parse_error = self._generate_from_plan_batched(
                request, system_chunks, hr_chunks, all_chunks
            )
        else:
            questions, parse_error = self._generate_from_plan_single(
                request, system_chunks, hr_chunks, all_chunks
            )

        audience = (getattr(request, "audience", None) or "jd_practice").strip().lower()
        if audience == "coach":
            for q in questions:
                q.image_hint = None

        return GenerateQuestionsFromPlanResponse(
            success=parse_error is None and len(questions) > 0,
            questions=questions,
            processing_time_ms=(time.time() - start) * 1000,
            error=parse_error,
        )

    def _generate_from_plan_single(
        self,
        request: GenerateQuestionsFromPlanRequest,
        system_chunks: list[RetrievedChunk],
        hr_chunks: list[RetrievedChunk],
        all_chunks: list[RetrievedChunk],
        *,
        order_start: int | None = None,
        order_end: int | None = None,
        batch_count: int | None = None,
    ) -> tuple[list, str | None]:
        user_message = self._build_candidate_from_plan_user_message(
            request,
            system_chunks,
            order_start=order_start,
            order_end=order_end,
            batch_count=batch_count,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": candidate_questions_system_prompt(request.language)},
            {"role": "user", "content": user_message},
        ]

        timeout = max(float(self._settings.request_timeout_seconds), 180.0)
        raw, llm_error = self._call_llm_safe(messages, timeout=timeout)
        if llm_error:
            return [], llm_error

        chunk_lookup = build_chunk_lookup(all_chunks)
        questions, parse_error = self._parse_questions_from_plan(
            raw,
            chunk_lookup,
            request.approved_plan,
            request.job_description,
            request.hr_note,
            expected_count=batch_count,
        )

        if parse_error and is_retryable_json_error(parse_error):
            raw, parse_error, questions = self._retry_parse(
                messages,
                raw,
                chunk_lookup,
                lambda r, lk: self._parse_questions_from_plan(
                    r,
                    lk,
                    request.approved_plan,
                    request.job_description,
                    request.hr_note,
                    expected_count=batch_count,
                ),
                timeout=timeout,
            )

        return questions, parse_error

    def _build_candidate_from_plan_user_message(
        self,
        request: GenerateQuestionsFromPlanRequest,
        system_chunks: list[RetrievedChunk],
        *,
        order_start: int | None = None,
        order_end: int | None = None,
        batch_count: int | None = None,
    ) -> str:
        plan_json = request.approved_plan.model_dump(by_alias=True, mode="json")
        target = batch_count or request.approved_plan.total_questions
        audience = getattr(request, "audience", None) or "jd_practice"
        cv = getattr(request, "cv_context", None)
        lines = [
            "[YÊU CẦU CANDIDATE]",
            f"audience: {audience}",
            f"ownerId: {request.owner_id}",
            f"jobDescription: {request.job_description}",
            language_instruction_block(request.language),
        ]
        if cv and str(cv).strip():
            lines.append(f"cvContext: {str(cv).strip()}")
        note = _effective_note(request)
        if note:
            lines.append(f"candidateNote: {note}")
        if order_start and order_end:
            lines.append(f"Sinh câu order {order_start}–{order_end} (batch {target} câu).")
        elif str(audience).strip().lower() == "coach":
            target = max(int(target or 0), 10)
            lines.append(
                "audience=coach: sinh TỐI THIỂU 10 câu; độ khó tăng dần "
                "(đầu easy → giữa medium → cuối hard). Không bịa skill ngoài cvContext."
            )
        lines.append("approvedPlan:")
        import json

        lines.append(json.dumps(plan_json, ensure_ascii=False))
        lines.append(
            "\nKhông dùng prompt HR. Không bắt image_hint / Mixed / code template. "
            "Thiếu chunk SYSTEM vẫn sinh đủ câu."
        )
        lines.append("\n" + format_retrieved_context(system_chunks, []))
        lines.append(f"\nTạo đúng {target} câu hỏi. Trả về JSON theo schema.")
        return "\n".join(lines)
