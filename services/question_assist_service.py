"""Ask AI cho từng câu hỏi — giữ ngữ cảnh JD/plan/citations (SCRUM-215)."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI

from config.settings import Settings
from models.internal_schemas import (
    QuestionAssistRequest,
    QuestionAssistResponse,
    QuestionAssistSuggestion,
)
from services.json_output_parser import build_json_fix_prompt, extract_json_object, is_retryable_json_error

logger = logging.getLogger(__name__)

QUESTION_ASSIST_SYSTEM_PROMPT = """Bạn là trợ lý AI cho HR Manager, hỗ trợ tinh chỉnh câu hỏi phỏng vấn đã sinh.

## Quy tắc bắt buộc
1. CHỈ trả lời trong phạm vi ngữ cảnh được cung cấp: JD, HR note, plan, câu hỏi hiện tại, citations.
2. Từ chối lịch sự nếu HR hỏi ngoài phạm vi (thời tiết, chính trị, chủ đề không liên quan JD/câu hỏi).
3. Trả lời bằng TIẾNG VIỆT.
4. Chỉ trả về JSON hợp lệ, không markdown fence, không text ngoài JSON.
5. Field `reply`: giải thích ngắn gọn, hữu ích cho HR.
6. Field `suggestion`: chỉ điền khi HR yêu cầu chỉnh sửa/cải thiện câu hỏi (khó hơn, behavioral, sample answer...).
   - Nếu HR chỉ hỏi giải thích/rationale/citation → `suggestion` = null.
   - Khi có suggestion, chỉ điền các field HR muốn thay đổi; field không đổi để null.
7. Không bịa citation hoặc yêu cầu không có trong JD/plan.

## Schema JSON
{
  "reply": "string",
  "suggestion": {
    "question": "string | null",
    "rationale": "string | null",
    "sampleAnswer": "string | null",
    "difficulty": "easy|medium|hard | null",
    "questionType": "technical|behavioral|situational|system-design|problem-solving | null"
  } | null
}
"""


class QuestionAssistService:
    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def assist(self, request: QuestionAssistRequest) -> QuestionAssistResponse:
        started = time.perf_counter()
        try:
            payload = self._build_user_payload(request)
            raw = self._call_llm(payload)
            parsed, err = extract_json_object(raw)
            if parsed is None and err and is_retryable_json_error(err):
                raw = self._call_llm(payload, fix_prompt=build_json_fix_prompt(raw))
                parsed, err = extract_json_object(raw)

            if parsed is None:
                return QuestionAssistResponse(
                    success=False,
                    error=err or "LLM không trả về JSON hợp lệ.",
                    processing_time_ms=self._elapsed_ms(started),
                )

            reply = str(parsed.get("reply") or "").strip()
            if not reply:
                return QuestionAssistResponse(
                    success=False,
                    error="LLM trả về reply rỗng.",
                    processing_time_ms=self._elapsed_ms(started),
                )

            suggestion = self._parse_suggestion(parsed.get("suggestion"))
            return QuestionAssistResponse(
                success=True,
                reply=reply,
                suggestion=suggestion,
                processing_time_ms=self._elapsed_ms(started),
            )
        except Exception as exc:
            logger.exception("question-assist failed")
            return QuestionAssistResponse(
                success=False,
                error="Xử lý Ask AI thất bại.",
                detail=str(exc),
                processing_time_ms=self._elapsed_ms(started),
            )

    def _build_user_payload(self, request: QuestionAssistRequest) -> str:
        ctx = request.context
        history = [{"role": m.role, "content": m.content} for m in request.chat_history]
        body = {
            "hrMessage": request.hr_message,
            "context": {
                "jobDescription": ctx.job_description,
                "hrNote": ctx.hr_note,
                "plan": ctx.plan,
                "difficulty": ctx.difficulty,
                "skills": ctx.skills,
                "currentQuestion": ctx.current_question,
                "isManuallyEdited": ctx.is_manually_edited,
            },
            "chatHistory": history,
        }
        return json.dumps(body, ensure_ascii=False, indent=2)

    def _call_llm(self, user_payload: str, *, fix_prompt: str | None = None) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": QUESTION_ASSIST_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ]
        if fix_prompt:
            messages.append({"role": "user", "content": fix_prompt})

        response = self._client.chat.completions.create(
            model=self._settings.chat_model,
            messages=messages,
            temperature=0.4,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _parse_suggestion(raw: Any) -> QuestionAssistSuggestion | None:
        if raw is None or not isinstance(raw, dict):
            return None

        suggestion = QuestionAssistSuggestion(
            question=raw.get("question"),
            rationale=raw.get("rationale"),
            sample_answer=raw.get("sampleAnswer") or raw.get("sample_answer"),
            difficulty=raw.get("difficulty"),
            question_type=raw.get("questionType") or raw.get("question_type"),
        )

        if not any(
            [
                suggestion.question,
                suggestion.rationale,
                suggestion.sample_answer,
                suggestion.difficulty,
                suggestion.question_type,
            ]
        ):
            return None
        return suggestion

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
