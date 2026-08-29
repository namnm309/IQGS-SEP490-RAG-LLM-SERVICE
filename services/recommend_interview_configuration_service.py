"""AI recommend interview configuration từ JD + job profile (+ optional KB retrieve)."""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from config.settings import Settings
from models.internal_schemas import (
    RecommendInterviewConfigurationRequest,
    RecommendInterviewConfigurationResponse,
)
from services.json_output_parser import (
    build_json_fix_prompt,
    extract_json_object,
    is_retryable_json_error,
)
from services.rag_context_helpers import format_retrieved_context
from services.rag_retrieval_service import RagRetrievalService
from services.recommend_configuration_validator import validate_and_normalize_recommendation

logger = logging.getLogger(__name__)

RECOMMEND_SYSTEM_PROMPT = """Bạn là chuyên gia thiết kế phỏng vấn tuyển dụng (interview configuration advisor).

## Nhiệm vụ
Đề xuất cấu hình phỏng vấn dựa trên Job Description, job profile đã trích xuất, và context KB (nếu có).
CHỈ đề xuất dựa trên JD/profile/KB — KHÔNG dùng template cố định theo role (vd. không luôn ASP.NET cho backend).

## Quy tắc bắt buộc
1. CHỈ trả JSON hợp lệ, không markdown fence.
2. Focus areas PHẢI bắt nguồn từ skills/responsibilities trong JD hoặc KB context.
3. sourceReason: giải thích bằng chứng JD/KB (KHÔNG cần trích nguyên văn). Ví dụ: "The JD explicitly requires PostgreSQL and database performance optimization."
4. questionDistribution: CHỈ 3 category lowercase: technical, behavioral, situational.
   - system_design và problem_solving là questionStyles, KHÔNG phải category riêng.
5. questionStyles: chỉ dùng: system_design, problem_solving, debugging, performance_analysis, coding, code_review, theory.
6. codingTaskTypes: chỉ BUG_DETECTION, CODE_COMPLETION, REFACTORING, TEST_CASE_DESIGN, PERFORMANCE_ANALYSIS.
   - KHÔNG dùng SYSTEM_DESIGN làm coding task (system_design là style).
7. difficulty: easy | medium | hard (lowercase).
8. Tổng questionDistribution.questionCount = numberOfQuestions; tổng percentage = 100.
9. Tổng focusAreas.weight = 100.

## Schema JSON
{
  "recommendedConfiguration": {
    "numberOfQuestions": 10,
    "difficulty": "hard",
    "questionDistribution": [
      {"category": "technical", "percentage": 60, "questionCount": 6},
      {"category": "behavioral", "percentage": 20, "questionCount": 2},
      {"category": "situational", "percentage": 20, "questionCount": 2}
    ],
    "focusAreas": [
      {
        "name": "ASP.NET Core",
        "weight": 25,
        "description": "Core framework skills for the role",
        "sourceReason": "The JD lists ASP.NET Core as a primary requirement.",
        "orderIndex": 0
      }
    ],
    "questionStyles": ["system_design", "problem_solving", "debugging"],
    "codingTaskTypes": ["BUG_DETECTION", "REFACTORING"],
    "codingTasksRecommended": true
  }
}
"""


class RecommendInterviewConfigurationService:
    def __init__(
        self,
        retrieval: RagRetrievalService,
        client: OpenAI,
        settings: Settings,
    ) -> None:
        self._retrieval = retrieval
        self._client = client
        self._settings = settings

    def recommend(
        self, request: RecommendInterviewConfigurationRequest
    ) -> RecommendInterviewConfigurationResponse:
        jd = (request.job_description or "").strip()
        if not jd:
            return RecommendInterviewConfigurationResponse(
                success=False,
                error="JD trống",
                detail="jobDescription không được rỗng.",
                stage="RECOMMEND_CONFIG",
                exception_type="EmptyJd",
                errors=["jobDescription không được rỗng."],
            )

        profile = request.job_profile.model_dump(by_alias=True) if request.job_profile else {}
        doc_ids = request.document_ids or []
        total_hint = request.number_of_questions or 10

        try:
            hr_chunks: list = []
            system_chunks: list = []
            if doc_ids:
                system_chunks, hr_chunks = self._retrieval.retrieve_for_job(
                    jd,
                    request.owner_id,
                    document_ids=[str(d) for d in doc_ids],
                )

            user_msg = self._build_user_message(jd, profile, system_chunks, hr_chunks, total_hint)
            raw = self._call_llm(user_msg)
            parsed, err = extract_json_object(raw)
            if parsed is None and err and is_retryable_json_error(err):
                raw = self._call_llm(user_msg, fix_prompt=build_json_fix_prompt(raw))
                parsed, err = extract_json_object(raw)

            if parsed is None:
                return RecommendInterviewConfigurationResponse(
                    success=False,
                    error="Không tạo được recommendation",
                    detail=err or "LLM không trả JSON hợp lệ.",
                    stage="RECOMMEND_CONFIG",
                    exception_type="LlmJsonError",
                    errors=[err or "LLM không trả JSON hợp lệ."],
                )

            normalized, val_errors = validate_and_normalize_recommendation(
                parsed, default_total=total_hint
            )
            if val_errors:
                return RecommendInterviewConfigurationResponse(
                    success=False,
                    error="Recommendation không hợp lệ",
                    detail="; ".join(val_errors),
                    stage="RECOMMEND_CONFIG",
                    exception_type="ValidationError",
                    errors=val_errors,
                )

            return RecommendInterviewConfigurationResponse(
                success=True,
                recommended_configuration=normalized,
            )
        except Exception as ex:
            logger.exception("recommend-interview-configuration failed")
            return RecommendInterviewConfigurationResponse(
                success=False,
                error="Recommend configuration thất bại",
                detail=str(ex),
                stage="RECOMMEND_CONFIG",
                exception_type=type(ex).__name__,
                errors=[str(ex)],
            )

    def _build_user_message(
        self,
        jd: str,
        profile: dict[str, Any],
        system_chunks: list,
        hr_chunks: list,
        total_hint: int,
    ) -> str:
        lines = [
            "[JOB PROFILE]",
            json.dumps(profile, ensure_ascii=False),
            "",
            f"[TARGET] numberOfQuestions hint: {total_hint}",
            "",
            "[JOB DESCRIPTION]",
            jd[:12000],
        ]
        if system_chunks or hr_chunks:
            lines.append("")
            lines.append(format_retrieved_context(system_chunks, hr_chunks))
        lines.append("")
        lines.append("Trả JSON recommendedConfiguration theo schema. Focus areas phải khác nhau theo JD.")
        return "\n".join(lines)

    def _call_llm(self, user_payload: str, *, fix_prompt: str | None = None) -> str:
        messages = [
            {"role": "system", "content": RECOMMEND_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ]
        if fix_prompt:
            messages.append({"role": "user", "content": fix_prompt})
        response = self._client.chat.completions.create(
            model=self._settings.chat_model,
            messages=messages,
            temperature=0.2,
        )
        return (response.choices[0].message.content or "").strip()
