"""Sinh AI Insight tổng quan phiên practice (SCRUM-305) — song ngữ + skillsToImprove."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI

from config.settings import Settings
from models.internal_schemas import (
    PracticeSessionInsightRequest,
    PracticeSessionInsightResponse,
)
from services.json_output_parser import build_json_fix_prompt, extract_json_object, is_retryable_json_error

logger = logging.getLogger(__name__)

MAX_QUESTIONS_IN_PROMPT = 10

PRACTICE_SESSION_INSIGHT_SYSTEM_PROMPT = """Bạn là giám khảo phỏng vấn AI, tổng hợp nhận xét cho cả phiên luyện tập của ứng viên.

## Quy tắc bắt buộc
1. Viết insightVi bằng TIẾNG VIỆT, insightEn bằng TIẾNG ANH — mỗi cái 1–2 câu, ngắn gọn, động viên + hướng cải thiện.
2. KHÔNG liệt kê từng câu hỏi; nhận xét tổng quan toàn phiên.
3. skillsToImproveVi và skillsToImproveEn: 2–5 kỹ năng/chủ đề ứng viên nên luyện thêm.
   - Ưu tiên vùng điểm thấp, skill tag, questionType, dimensionScores yếu.
   - Không bịa kỹ năng ngoài context bộ câu hỏi / setSkills.
   - Tên kỹ năng ngắn gọn (vd. "C# / .NET", "Giao tiếp kỹ thuật").
4. Nếu answeredCount = 0 hoặc answeredCount << totalQuestions: insight nên nhắc hoàn thành nhiều câu hơn;
   skillsToImprove lấy từ setSkills hoặc phân bố questionType của các câu đã chấm / còn lại.
5. Chỉ trả về JSON hợp lệ, không markdown fence, không text ngoài JSON.

## Schema JSON
{
  "insightVi": "string",
  "insightEn": "string",
  "skillsToImproveVi": ["string"],
  "skillsToImproveEn": ["string"]
}
"""


class PracticeSessionInsightService:
    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def generate(self, request: PracticeSessionInsightRequest) -> PracticeSessionInsightResponse:
        started = time.perf_counter()
        try:
            if request.total_questions <= 0:
                return PracticeSessionInsightResponse(
                    success=False,
                    error="totalQuestions phải lớn hơn 0.",
                    processing_time_ms=self._elapsed_ms(started),
                )

            payload = self._build_user_payload(request)
            raw = self._call_llm(payload)
            parsed, err = extract_json_object(raw)
            if parsed is None and err and is_retryable_json_error(err):
                raw = self._call_llm(payload, fix_prompt=build_json_fix_prompt(raw))
                parsed, err = extract_json_object(raw)

            if parsed is None:
                return PracticeSessionInsightResponse(
                    success=False,
                    error=err or "LLM không trả về JSON hợp lệ.",
                    processing_time_ms=self._elapsed_ms(started),
                )

            insight_vi = self._parse_optional_str(parsed.get("insightVi") or parsed.get("insight_vi"))
            insight_en = self._parse_optional_str(parsed.get("insightEn") or parsed.get("insight_en"))
            if not insight_vi or not insight_en:
                return PracticeSessionInsightResponse(
                    success=False,
                    error="LLM thiếu insightVi hoặc insightEn.",
                    processing_time_ms=self._elapsed_ms(started),
                )

            skills_vi = self._parse_string_list(
                parsed.get("skillsToImproveVi") or parsed.get("skills_to_improve_vi")
            )
            skills_en = self._parse_string_list(
                parsed.get("skillsToImproveEn") or parsed.get("skills_to_improve_en")
            )
            # Đồng bộ độ dài — lấy min 2–5 nếu có dữ liệu
            skills_vi = skills_vi[:5]
            skills_en = skills_en[:5]

            return PracticeSessionInsightResponse(
                success=True,
                insight_vi=insight_vi,
                insight_en=insight_en,
                skills_to_improve_vi=skills_vi,
                skills_to_improve_en=skills_en,
                processing_time_ms=self._elapsed_ms(started),
            )
        except Exception as exc:
            logger.exception("practice-session-insight failed")
            return PracticeSessionInsightResponse(
                success=False,
                error="Sinh AI Insight thất bại.",
                detail=str(exc),
                processing_time_ms=self._elapsed_ms(started),
            )

    def _build_user_payload(self, request: PracticeSessionInsightRequest) -> str:
        # Ưu tiên câu điểm thấp đưa vào prompt (giới hạn token)
        summaries = sorted(
            request.question_summaries,
            key=lambda q: (q.score is None, q.score if q.score is not None else 999),
        )[:MAX_QUESTIONS_IN_PROMPT]

        body = {
            "overallScore": request.overall_score,
            "totalQuestions": request.total_questions,
            "answeredCount": request.answered_count,
            "setTitle": request.set_title,
            "setSkills": request.set_skills,
            "questionSummaries": [
                {
                    "questionType": q.question_type,
                    "skill": q.skill,
                    "score": q.score,
                    "strengths": q.strengths[:2],
                    "improvements": q.improvements[:2],
                    "dimensionScores": q.dimension_scores,
                }
                for q in summaries
            ],
        }
        return json.dumps(body, ensure_ascii=False, indent=2)

    def _call_llm(self, user_payload: str, *, fix_prompt: str | None = None) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": PRACTICE_SESSION_INSIGHT_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ]
        if fix_prompt:
            messages.append({"role": "user", "content": fix_prompt})

        response = self._client.chat.completions.create(
            model=self._settings.chat_model,
            messages=messages,
            temperature=self._settings.temperature,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _parse_string_list(raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        result: list[str] = []
        for item in raw:
            text = str(item or "").strip()
            if text:
                result.append(text)
        return result

    @staticmethod
    def _parse_optional_str(raw: Any) -> str | None:
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
