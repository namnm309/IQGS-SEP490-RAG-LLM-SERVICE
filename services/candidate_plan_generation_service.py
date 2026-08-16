"""Sinh plan luyện tập cho Candidate — SYSTEM retrieve only, prompt không phải HR."""
from __future__ import annotations

import logging
import time

from models.internal_schemas import GeneratePlanRequest, GeneratePlanResponse
from services.candidate_generation_prompts import candidate_plan_system_prompt
from services.plan_generation_service import PlanGenerationService
from services.rag_context_helpers import format_retrieved_context
from helpers.language_prompt import language_instruction_block
from vectorstores.base import RetrievedChunk

logger = logging.getLogger(__name__)


def _effective_jd(request: GeneratePlanRequest) -> str:
    cv = getattr(request, "cv_context", None)
    if cv and str(cv).strip():
        return str(cv).strip()
    return (request.job_description or "").strip()


def _effective_note(request: GeneratePlanRequest) -> str | None:
    note = getattr(request, "candidate_note", None)
    if note and str(note).strip():
        return str(note).strip()
    if request.hr_note and request.hr_note.strip():
        return request.hr_note.strip()
    return None


class CandidatePlanGenerationService(PlanGenerationService):
    def generate(self, request: GeneratePlanRequest) -> GeneratePlanResponse:
        start = time.time()
        if not request.owner_id or not request.owner_id.strip():
            return self._fail(start, "ownerId là bắt buộc", "VALIDATION")

        jd = _effective_jd(request)
        if not jd:
            return self._fail(
                start, "cvContext hoặc jobDescription là bắt buộc", "VALIDATION"
            )

        # Parse/citation HR vẫn cần job_description không rỗng
        if not (request.job_description or "").strip():
            request.job_description = jd

        note = _effective_note(request)
        if note:
            request.hr_note = note

        system_chunks: list[RetrievedChunk] = []
        try:
            retrieve_sys = getattr(self._retrieval, "retrieve_system_only", None)
            if retrieve_sys is not None:
                system_chunks = retrieve_sys(jd, query_extra=note) or []
            else:
                system_chunks, _hr = self._retrieval.retrieve_for_job(
                    jd, request.owner_id.strip(), query_extra=note
                )
                system_chunks = system_chunks or []
        except Exception as exc:
            logger.warning("Candidate plan SYSTEM retrieve thất bại — vẫn sinh plan: %s", exc)

        user_message = self._build_candidate_user_message(request, system_chunks)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": candidate_plan_system_prompt(request.language)},
            {"role": "user", "content": user_message},
        ]

        try:
            raw = self._call_llm(messages)
        except Exception as exc:
            logger.exception("Candidate plan LLM failed")
            return self._fail(start, "Lỗi sinh plan", "PLAN_GENERATION", f"Lỗi LLM: {exc}")

        from services.rag_context_helpers import build_chunk_lookup
        from services.json_output_parser import (
            build_json_fix_prompt,
            build_plan_validation_fix_prompt,
            is_retryable_plan_error,
        )

        chunk_lookup = build_chunk_lookup(system_chunks)
        plan, parse_error = self._parse_plan(raw, chunk_lookup, request)

        if parse_error and is_retryable_plan_error(parse_error):
            fix_prompt = (
                build_json_fix_prompt(raw)
                if parse_error.startswith(("LLM không", "Không parse", "JSON thiếu", "Không map"))
                else build_plan_validation_fix_prompt(parse_error, raw)
            )
            retry_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": fix_prompt},
            ]
            try:
                raw = self._call_llm(retry_messages)
            except Exception as exc:
                logger.exception("Candidate plan LLM retry failed")
                return self._fail(
                    start, "Lỗi sinh plan", "PLAN_GENERATION", f"Lỗi LLM khi retry: {exc}"
                )
            plan, parse_error = self._parse_plan(raw, chunk_lookup, request)

        if parse_error:
            return self._fail(start, "Lỗi sinh plan", "PLAN_GENERATION", parse_error)

        return GeneratePlanResponse(
            success=True,
            plan=plan,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def _build_candidate_user_message(
        self,
        request: GeneratePlanRequest,
        system_chunks: list[RetrievedChunk],
    ) -> str:
        audience = getattr(request, "audience", None) or "jd_practice"
        cv = getattr(request, "cv_context", None)
        skills_text = ", ".join(request.skills) if request.skills else ""
        types_text = ", ".join(request.question_types)
        lines = [
            "[YÊU CẦU CANDIDATE]",
            f"audience: {audience}",
            f"ownerId: {request.owner_id}",
            f"jobDescription: {request.job_description}",
            f"so_cau: {request.number_of_questions}",
            f"difficulty: {request.difficulty}",
            f"loai_cau: {types_text}",
            language_instruction_block(request.language),
        ]
        if cv and str(cv).strip():
            lines.append(f"cvContext: {str(cv).strip()}")
        if skills_text:
            lines.append(f"skills: {skills_text}")
        note = _effective_note(request)
        if note:
            lines.append(f"candidateNote: {note}")
        lines.append(
            "\nKhông dùng prompt HR. Không bắt citation JD / image_hint / code template. "
            "Context [HỆ THỐNG] phụ, có thể trống."
        )
        lines.append("\n" + format_retrieved_context(system_chunks, []))
        lines.append("\nTạo plan JSON theo schema. Không sinh câu hỏi cụ thể.")
        return "\n".join(lines)
