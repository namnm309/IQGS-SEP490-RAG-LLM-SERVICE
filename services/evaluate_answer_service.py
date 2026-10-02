"""Đánh giá câu trả lời Candidate theo rubric (SCRUM-281) + Coach dimensions (SCRUM-447)."""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI

from config.settings import Settings
from models.internal_schemas import EvaluateAnswerRequest, EvaluateAnswerResponse
from helpers.language_prompt import language_instruction_block
from services.json_output_parser import build_json_fix_prompt, extract_json_object, is_retryable_json_error

logger = logging.getLogger(__name__)

EVALUATE_ANSWER_SYSTEM_PROMPT = """Bạn là giám khảo phỏng vấn AI, chấm điểm câu trả lời của ứng viên.

## Quy tắc bắt buộc
1. Chấm dựa trên câu hỏi, tiêu chí đánh giá (evaluationCriteria), và câu trả lời ứng viên.
2. sampleAnswer (nếu có) chỉ dùng nội bộ để đối chiếu — KHÔNG nhắc đến sample answer trong strengths/improvements/suggestion.
3. Viết mọi trường chữ tự do theo OUTPUT_LANGUAGE ở cuối prompt.
4. Chỉ trả về JSON hợp lệ, không markdown fence, không text ngoài JSON.
5. score là số thực từ 0 đến 100 (có thể có phần thập phân).
6. strengths: 1–4 điểm mạnh cụ thể của câu trả lời.
7. improvements: 1–4 điểm cần cải thiện cụ thể.
8. suggestion: 1–2 câu gợi ý cải thiện ngắn gọn.
9. dimensionScores (optional): điểm theo chiều (ví dụ clarity, depth, structure, relevance) — mỗi chiều 0–100.
   Nếu không đủ thông tin thì để null.

## Schema JSON
{
  "score": 0-100,
  "strengths": ["string"],
  "improvements": ["string"],
  "suggestion": "string",
  "dimensionScores": { "clarity": 0-100, "depth": 0-100 } | null
}
"""

# SCRUM-447: Coach assessment — LLM chỉ chấm dimensions; Backend tính AnswerScore/SkillScore.
EVALUATE_COACH_SYSTEM_PROMPT = """Bạn là giám khảo Coach competency. Chỉ chấm 3 chiều — KHÔNG kết luận Fresher/Junior/Middle/Senior.

## Quy tắc bắt buộc
1. Chỉ JSON hợp lệ. Viết strengths, improvements và suggestion theo OUTPUT_LANGUAGE ở cuối prompt.
2. BẮT BUỘC dimensionScores với đúng 3 key: correctness, relevance, clarity (mỗi chiều 0–100).
3. score (optional): có thể để null — Backend sẽ tính overall. Nếu có score thì chỉ là gợi ý hiển thị, không thay công thức Backend.
4. strengths / improvements / suggestion như bình thường.
5. sampleAnswer chỉ dùng nội bộ — không nhắc trong feedback.

## Schema JSON
{
  "score": null,
  "strengths": ["string"],
  "improvements": ["string"],
  "suggestion": "string",
  "dimensionScores": {
    "correctness": 0-100,
    "relevance": 0-100,
    "clarity": 0-100
  }
}
"""


class EvaluateAnswerService:
    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def evaluate(self, request: EvaluateAnswerRequest) -> EvaluateAnswerResponse:
        started = time.perf_counter()
        try:
            if not (request.candidate_answer or "").strip():
                return EvaluateAnswerResponse(
                    success=False,
                    error="Câu trả lời ứng viên không được để trống.",
                    processing_time_ms=self._elapsed_ms(started),
                )
            if not (request.question or "").strip():
                return EvaluateAnswerResponse(
                    success=False,
                    error="Câu hỏi không được để trống.",
                    processing_time_ms=self._elapsed_ms(started),
                )

            is_coach = (request.scoring_mode or "").strip().lower() == "coach"
            system_prompt = EVALUATE_COACH_SYSTEM_PROMPT if is_coach else EVALUATE_ANSWER_SYSTEM_PROMPT
            system_prompt = system_prompt + "\n\n" + language_instruction_block(request.language)

            payload = self._build_user_payload(request)
            raw = self._call_llm(payload, system_prompt=system_prompt)
            parsed, err = extract_json_object(raw)
            if parsed is None and err and is_retryable_json_error(err):
                raw = self._call_llm(
                    payload,
                    system_prompt=system_prompt,
                    fix_prompt=build_json_fix_prompt(raw),
                )
                parsed, err = extract_json_object(raw)

            if parsed is None:
                return EvaluateAnswerResponse(
                    success=False,
                    error=err or "LLM không trả về JSON hợp lệ.",
                    processing_time_ms=self._elapsed_ms(started),
                )

            dimension_scores = self._parse_dimension_scores(
                parsed.get("dimensionScores") or parsed.get("dimension_scores")
            )
            if is_coach:
                dimension_scores = self._ensure_coach_dimensions(dimension_scores, parsed)

            score = self._parse_score(parsed.get("score"))
            if score is None and is_coach and dimension_scores:
                # UI fallback — Backend Coach vẫn tính lại từ dimensions
                score = sum(dimension_scores.values()) / len(dimension_scores)
            if score is None and not is_coach:
                return EvaluateAnswerResponse(
                    success=False,
                    error="LLM trả về score không hợp lệ.",
                    processing_time_ms=self._elapsed_ms(started),
                )
            if is_coach and not dimension_scores:
                return EvaluateAnswerResponse(
                    success=False,
                    error="Coach evaluate thiếu dimensionScores (correctness/relevance/clarity).",
                    processing_time_ms=self._elapsed_ms(started),
                )

            return EvaluateAnswerResponse(
                success=True,
                score=score,
                strengths=self._parse_string_list(parsed.get("strengths")),
                improvements=self._parse_string_list(parsed.get("improvements")),
                suggestion=self._parse_optional_str(parsed.get("suggestion")),
                dimension_scores=dimension_scores,
                processing_time_ms=self._elapsed_ms(started),
            )
        except Exception as exc:
            logger.exception("evaluate-answer failed")
            return EvaluateAnswerResponse(
                success=False,
                error="Đánh giá câu trả lời thất bại.",
                detail=str(exc),
                processing_time_ms=self._elapsed_ms(started),
            )

    def _build_user_payload(self, request: EvaluateAnswerRequest) -> str:
        body = {
            "question": request.question,
            "evaluationCriteria": request.evaluation_criteria,
            "candidateAnswer": request.candidate_answer,
            "sampleAnswer": request.sample_answer,
            "jdContext": request.jd_context,
            "skill": request.skill,
            "questionType": request.question_type,
            "scoringMode": request.scoring_mode,
        }
        return json.dumps(body, ensure_ascii=False, indent=2)

    def _call_llm(
        self,
        user_payload: str,
        *,
        system_prompt: str = EVALUATE_ANSWER_SYSTEM_PROMPT,
        fix_prompt: str | None = None,
    ) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
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

    @classmethod
    def _ensure_coach_dimensions(
        cls, dims: dict[str, float] | None, parsed: dict[str, Any]
    ) -> dict[str, float] | None:
        """Chuẩn hóa aliases → correctness/relevance/clarity."""
        source = dict(dims or {})
        # Một số model trả key riêng ngoài dimensionScores
        for key in ("correctness", "relevance", "clarity", "accuracy"):
            if key in parsed and key not in source:
                try:
                    source[key] = max(0.0, min(100.0, float(parsed[key])))
                except (TypeError, ValueError):
                    pass

        def pick(*names: str) -> float | None:
            for n in names:
                for k, v in source.items():
                    if k.lower() == n.lower():
                        return float(v)
            return None

        correctness = pick("correctness", "accuracy")
        relevance = pick("relevance")
        clarity = pick("clarity")
        if correctness is None or relevance is None or clarity is None:
            return None
        return {
            "correctness": correctness,
            "relevance": relevance,
            "clarity": clarity,
        }

    @staticmethod
    def _parse_score(raw: Any) -> float | None:
        if raw is None:
            return None
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(100.0, score))

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
    def _parse_dimension_scores(raw: Any) -> dict[str, float] | None:
        if not isinstance(raw, dict) or not raw:
            return None
        result: dict[str, float] = {}
        for key, value in raw.items():
            name = str(key or "").strip()
            if not name:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            result[name] = max(0.0, min(100.0, score))
        return result or None

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
