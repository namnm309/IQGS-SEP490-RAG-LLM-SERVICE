"""Đánh giá độ phù hợp bộ câu hỏi với JD — chỉ nhận xét lời, không điểm số."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI

from config.settings import Settings
from models.internal_schemas import (
    EvaluateQuestionSetRequest,
    EvaluateQuestionSetResponse,
    JD_FIT_ACTION_TYPES,
    JD_FIT_FLAGS,
    JD_FIT_VERDICTS,
    JdFitQuestionFlagItem,
    JdFitSourceItem,
    JdFitSuggestedActionItem,
)
from services.json_output_parser import build_json_fix_prompt, extract_json_object, is_retryable_json_error
from services.rag_context_helpers import (
    _split_jd_units,
    match_jd_unit_index,
    select_relevant_jd_excerpt,
)

logger = logging.getLogger(__name__)

_MAX_JD_UNITS = 24
_MAX_OVERALL_SOURCES = 8
_MAX_FLAG_SOURCES = 3
_EXCERPT_CAP = 400

EVALUATE_QUESTION_SET_SYSTEM_PROMPT = """Bạn là chuyên gia tuyển dụng, đánh giá bộ câu hỏi phỏng vấn so với Job Description.

## Quy tắc bắt buộc
1. Chỉ nhận xét bằng LỜI — CẤM mọi điểm số, phần trăm, thang 0–100, xếp hạng số.
2. verdict chỉ được một trong: unfit | fair | good | excellent
   - unfit: lệch JD rõ, thiếu phần bắt buộc hoặc nhiều câu lạc đề
   - fair: cover một phần; còn thiếu skill/loại câu quan trọng
   - good: bám JD, thiếu vài chỗ nhỏ
   - excellent: đúng role/level, skill JD được cover, gần như không câu lạc đề
3. summaryVi bằng TIẾNG VIỆT, summaryEn bằng TIẾNG ANH — mỗi cái 1–2 câu, không số.
4. questionFlags.flag chỉ: onJd | weak | offJd | duplicate
5. suggestedActions.type chỉ: add | rewrite | remove
6. jdSources và questionFlags.sources CHỈ được dùng chunkIndex có trong jdChunks. KHÔNG bịa excerpt, KHÔNG paraphrase — chỉ trả chunkIndex.
7. jdSources: 1–8 index dùng cho verdict/tóm tắt. Mỗi questionFlag: 1–3 sources.
8. Chỉ trả JSON hợp lệ, không markdown fence.

## Schema JSON
{
  "verdict": "unfit|fair|good|excellent",
  "summaryVi": "string",
  "summaryEn": "string",
  "jdSources": [{ "chunkIndex": 0 }],
  "questionFlags": [
    {
      "questionId": "string",
      "order": 1,
      "flag": "onJd|weak|offJd|duplicate",
      "noteVi": "string",
      "noteEn": "string",
      "sources": [{ "chunkIndex": 0 }]
    }
  ],
  "missingTopics": ["string"],
  "suggestedActions": [
    { "type": "add|rewrite|remove", "questionId": "string|null", "reasonVi": "string", "reasonEn": "string" }
  ]
}
"""

_VERDICT_ALIASES = {
    "unfit": "unfit",
    "not a fit": "unfit",
    "chua phu hop": "unfit",
    "chưa phù hợp": "unfit",
    "poor": "unfit",
    "fair": "fair",
    "tuong doi": "fair",
    "tương đối": "fair",
    "partial": "fair",
    "good": "good",
    "tot": "good",
    "tốt": "good",
    "excellent": "excellent",
    "tuyet voi": "excellent",
    "tuyệt vời": "excellent",
    "great": "excellent",
}

_NUMERIC_SCORE_RE = re.compile(
    r"(?:\b\d{1,3}\s*/\s*100\b)|(?:\b\d{1,3}\s*%)|(?:\bscore\s*[:=]\s*\d+)",
    re.IGNORECASE,
)
_BARE_PERCENT_OR_SCORE = re.compile(r"\b\d{1,3}\s*%|\b\d{1,3}\s*/\s*100\b")

_FLAG_ALIASES = {
    "onjd": "onJd",
    "on_jd": "onJd",
    "on-jd": "onJd",
    "fit": "onJd",
    "weak": "weak",
    "hoi yeu": "weak",
    "offjd": "offJd",
    "off_jd": "offJd",
    "off-jd": "offJd",
    "duplicate": "duplicate",
    "trung": "duplicate",
}


class EvaluateQuestionSetService:
    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def evaluate(self, request: EvaluateQuestionSetRequest) -> EvaluateQuestionSetResponse:
        started = time.perf_counter()
        try:
            if not (request.job_description or "").strip():
                return EvaluateQuestionSetResponse(
                    success=False,
                    error="Job description không được để trống.",
                    processing_time_ms=self._elapsed_ms(started),
                )
            if not request.questions:
                return EvaluateQuestionSetResponse(
                    success=False,
                    error="Bộ câu hỏi chưa có câu hỏi.",
                    processing_time_ms=self._elapsed_ms(started),
                )

            units = self._jd_units(request.job_description)
            payload = self._build_user_payload(request, units)
            raw = self._call_llm(payload)
            parsed, err = extract_json_object(raw)
            if parsed is None and err and is_retryable_json_error(err):
                raw = self._call_llm(payload, fix_prompt=build_json_fix_prompt(raw))
                parsed, err = extract_json_object(raw)

            if parsed is None:
                return EvaluateQuestionSetResponse(
                    success=False,
                    error=err or "LLM không trả về JSON hợp lệ.",
                    processing_time_ms=self._elapsed_ms(started),
                )

            verdict = self._normalize_verdict(parsed.get("verdict"))
            summary_vi = self._strip_numeric(self._parse_optional_str(parsed.get("summaryVi") or parsed.get("summary_vi")))
            summary_en = self._strip_numeric(self._parse_optional_str(parsed.get("summaryEn") or parsed.get("summary_en")))
            if not verdict or not summary_vi or not summary_en:
                return EvaluateQuestionSetResponse(
                    success=False,
                    error="LLM thiếu verdict hoặc summaryVi/summaryEn.",
                    processing_time_ms=self._elapsed_ms(started),
                )

            flags = self._parse_flags(
                parsed.get("questionFlags") or parsed.get("question_flags"),
                units=units,
                questions=request.questions,
                job_description=request.job_description,
            )
            jd_sources = self._resolve_sources(
                parsed.get("jdSources") or parsed.get("jd_sources"),
                units,
                max_n=_MAX_OVERALL_SOURCES,
            )
            if not jd_sources:
                jd_sources = self._fallback_overall_sources(units, request.questions, request.job_description)

            return EvaluateQuestionSetResponse(
                success=True,
                verdict=verdict,
                summary_vi=summary_vi,
                summary_en=summary_en,
                question_flags=flags,
                missing_topics=self._parse_string_list(parsed.get("missingTopics") or parsed.get("missing_topics")),
                suggested_actions=self._parse_actions(
                    parsed.get("suggestedActions") or parsed.get("suggested_actions")
                ),
                jd_sources=jd_sources,
                processing_time_ms=self._elapsed_ms(started),
            )
        except Exception as exc:
            logger.exception("evaluate-question-set failed")
            return EvaluateQuestionSetResponse(
                success=False,
                error="Đánh giá bộ câu hỏi với JD thất bại.",
                detail=str(exc),
                processing_time_ms=self._elapsed_ms(started),
            )

    def _build_user_payload(self, request: EvaluateQuestionSetRequest, units: list[str]) -> str:
        questions = []
        for q in request.questions:
            questions.append(
                {
                    "questionId": q.question_id,
                    "order": q.order,
                    "question": q.question,
                    "questionType": q.question_type,
                    "difficulty": q.difficulty,
                    "skill": q.skill,
                    "focusArea": q.focus_area,
                    "rationale": q.rationale,
                }
            )
        body = {
            "jobDescription": request.job_description,
            "hrNote": request.hr_note,
            "setTitle": request.set_title,
            "plan": request.plan,
            "jdChunks": [
                {"chunkIndex": i, "excerpt": unit[:_EXCERPT_CAP]}
                for i, unit in enumerate(units)
            ],
            "questions": questions,
        }
        return json.dumps(body, ensure_ascii=False, indent=2)

    def _call_llm(self, user_payload: str, *, fix_prompt: str | None = None) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": EVALUATE_QUESTION_SET_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ]
        if fix_prompt:
            messages.append({"role": "user", "content": fix_prompt})

        response = self._client.chat.completions.create(
            model=self._settings.chat_model,
            messages=messages,
            temperature=getattr(self._settings, "temperature", 0.2),
        )
        return (response.choices[0].message.content or "").strip()

    @classmethod
    def _normalize_verdict(cls, raw: Any) -> str | None:
        key = str(raw or "").strip().lower()
        if not key:
            return None
        mapped = _VERDICT_ALIASES.get(key, key)
        return mapped if mapped in JD_FIT_VERDICTS else None

    @classmethod
    def _normalize_flag(cls, raw: Any) -> str | None:
        key = str(raw or "").strip().lower().replace(" ", "")
        mapped = _FLAG_ALIASES.get(key, raw if isinstance(raw, str) else None)
        if mapped in JD_FIT_FLAGS:
            return mapped
        return None

    def _parse_flags(
        self,
        raw: Any,
        *,
        units: list[str],
        questions: list,
        job_description: str,
    ) -> list[JdFitQuestionFlagItem]:
        if not isinstance(raw, list):
            return []
        by_id = {str(q.question_id): q for q in questions if q.question_id}
        result: list[JdFitQuestionFlagItem] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            flag = self._normalize_flag(item.get("flag") or item.get("flags"))
            if isinstance(item.get("flags"), list) and not flag:
                flag = self._normalize_flag(item["flags"][0] if item["flags"] else None)
            if not flag:
                continue
            qid = self._parse_optional_str(item.get("questionId") or item.get("question_id"))
            sources = self._resolve_sources(
                item.get("sources") or item.get("jdSources"),
                units,
                max_n=_MAX_FLAG_SOURCES,
            )
            if not sources:
                q = by_id.get(qid or "")
                hint = " ".join(
                    part
                    for part in (
                        q.question if q else None,
                        q.skill if q else None,
                        q.focus_area if q else None,
                    )
                    if part
                )
                sources = self._fallback_flag_sources(units, job_description, hint)
            result.append(
                JdFitQuestionFlagItem(
                    question_id=qid,
                    order=self._parse_optional_int(item.get("order")),
                    flag=flag,
                    note_vi=self._strip_numeric(self._parse_optional_str(item.get("noteVi") or item.get("note_vi") or item.get("note"))),
                    note_en=self._strip_numeric(self._parse_optional_str(item.get("noteEn") or item.get("note_en"))),
                    sources=sources,
                )
            )
        return result

    @classmethod
    def _jd_units(cls, job_description: str) -> list[str]:
        units = [u.strip() for u in _split_jd_units(job_description) if (u or "").strip()]
        return units[:_MAX_JD_UNITS]

    @classmethod
    def _resolve_sources(cls, raw: Any, units: list[str], *, max_n: int) -> list[JdFitSourceItem]:
        if not units:
            return []
        indexes: list[int] = []
        if isinstance(raw, list):
            for item in raw:
                idx: int | None = None
                if isinstance(item, dict):
                    idx = cls._parse_optional_int(
                        item.get("chunkIndex") if "chunkIndex" in item else item.get("chunk_index")
                    )
                elif isinstance(item, int):
                    idx = item
                elif isinstance(item, str) and item.strip().isdigit():
                    idx = int(item.strip())
                if idx is None or idx < 0 or idx >= len(units):
                    continue
                if idx not in indexes:
                    indexes.append(idx)
                if len(indexes) >= max_n:
                    break
        return [
            JdFitSourceItem(chunk_index=i, excerpt=units[i][:_EXCERPT_CAP])
            for i in indexes
        ]

    @classmethod
    def _fallback_flag_sources(cls, units: list[str], job_description: str, hint: str) -> list[JdFitSourceItem]:
        excerpt = select_relevant_jd_excerpt(job_description, hint or None)
        idx = match_jd_unit_index(units, excerpt)
        if idx is None:
            return []
        return [JdFitSourceItem(chunk_index=idx, excerpt=units[idx][:_EXCERPT_CAP])]

    @classmethod
    def _fallback_overall_sources(
        cls, units: list[str], questions: list, job_description: str
    ) -> list[JdFitSourceItem]:
        seen: set[int] = set()
        result: list[JdFitSourceItem] = []
        hints = [" ".join(filter(None, [q.question, q.skill, q.focus_area])) for q in questions[:_MAX_OVERALL_SOURCES]]
        if not hints:
            hints = [job_description[:120]]
        for hint in hints:
            for item in cls._fallback_flag_sources(units, job_description, hint):
                if item.chunk_index in seen:
                    continue
                seen.add(item.chunk_index)
                result.append(item)
                if len(result) >= _MAX_OVERALL_SOURCES:
                    return result
        if not result and units:
            result.append(JdFitSourceItem(chunk_index=0, excerpt=units[0][:_EXCERPT_CAP]))
        return result

    def _parse_actions(self, raw: Any) -> list[JdFitSuggestedActionItem]:
        if not isinstance(raw, list):
            return []
        result: list[JdFitSuggestedActionItem] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            action_type = str(item.get("type") or item.get("action") or "").strip().lower()
            if action_type not in JD_FIT_ACTION_TYPES:
                continue
            result.append(
                JdFitSuggestedActionItem(
                    type=action_type,
                    question_id=self._parse_optional_str(item.get("questionId") or item.get("question_id")),
                    reason_vi=self._strip_numeric(self._parse_optional_str(item.get("reasonVi") or item.get("reason_vi") or item.get("reason"))),
                    reason_en=self._strip_numeric(self._parse_optional_str(item.get("reasonEn") or item.get("reason_en"))),
                )
            )
        return result

    @staticmethod
    def _parse_string_list(raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        result: list[str] = []
        for item in raw:
            text = str(item or "").strip()
            if text:
                result.append(EvaluateQuestionSetService._strip_numeric(text) or text)
        return result

    @staticmethod
    def _parse_optional_str(raw: Any) -> str | None:
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    @staticmethod
    def _parse_optional_int(raw: Any) -> int | None:
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _strip_numeric(text: str | None) -> str | None:
        if not text:
            return text
        cleaned = _NUMERIC_SCORE_RE.sub("", text)
        cleaned = _BARE_PERCENT_OR_SCORE.sub("", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;.-")
        return cleaned or None

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 2)
