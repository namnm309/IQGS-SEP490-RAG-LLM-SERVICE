"""SCRUM-466: LLM classify documentType + isItRole — dùng chung CV / KB / JD.

Tách khỏi prompt JD-only để CV/KB không bị ép documentType=job_description.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI

from config.settings import Settings
from helpers.jd_validator import validate_it_domain
from services.json_output_parser import (
    build_json_fix_prompt,
    extract_json_object,
    is_retryable_json_error,
)

logger = logging.getLogger(__name__)

Purpose = Literal["cv", "kb", "jd"]

ALLOWED_DOCUMENT_TYPES = frozenset(
    {"job_description", "resume", "article", "documentation", "other"}
)

CLASSIFY_SYSTEM_PROMPT = """Bạn là trợ lý phân loại văn bản phục vụ hệ thống tuyển dụng IT.

## Nhiệm vụ
Phân loại văn bản: documentType + isItRole + rejectReason.

## Quy tắc (BẮT BUỘC)
1. documentType chỉ một trong:
   - job_description: đang TUYỂN NGƯỜI — vị trí + việc phải làm + yêu cầu.
   - resume: CV / hồ sơ ứng viên (kinh nghiệm cá nhân).
   - article: blog / tutorial / bài viết kỹ thuật (dạy / giải thích).
   - documentation: README / API docs / tài liệu kỹ thuật / policy nội bộ kỹ thuật / rubric / roadmap.
   - other: không thuộc các loại trên (hóa đơn, SOP marketing thuần, v.v.).
2. isItRole = true chỉ khi nội dung CHÍNH thuộc IT/phần mềm
   (dev, QA, DevOps, data, BA/PM kỹ thuật, security, cloud, stack nội bộ…).
3. Marketing/Sales/Kế toán/Luật dù có Excel/SQL → isItRole = false.
4. rejectReason: 1 câu tiếng Việt giải thích vì sao reject; null nếu pass theo ngữ cảnh.

## Schema JSON
{
  "documentType": "job_description|resume|article|documentation|other",
  "isItRole": true,
  "rejectReason": "string | null"
}
"""

_PASS_TYPES: dict[Purpose, frozenset[str]] = {
    "jd": frozenset({"job_description"}),
    "cv": frozenset({"resume"}),
    "kb": frozenset({"documentation", "article"}),
}

_DEFAULT_REJECT: dict[Purpose, str] = {
    "jd": (
        "Đây không phải tin tuyển dụng (Job Description). "
        "Vui lòng dán/upload JD đang tuyển vị trí IT/phần mềm."
    ),
    "cv": (
        "Đây không phải CV/hồ sơ ứng viên IT. "
        "Vui lòng tải lên CV kỹ thuật/phần mềm."
    ),
    "kb": (
        "Tài liệu không thuộc lĩnh vực IT/phần mềm. "
        "Hệ thống chỉ nhận tài liệu kỹ thuật phục vụ phỏng vấn IT."
    ),
}

_DEFAULT_NOT_IT = (
    "Nội dung không thuộc lĩnh vực IT/phần mềm. "
    "Hệ thống chỉ nhận tài liệu kỹ thuật/phần mềm."
)


@dataclass
class DocumentClassifyResult:
    success: bool
    document_type: str | None = None
    is_it_role: bool | None = None
    reject_reason: str | None = None
    error: str | None = None
    detail: str | None = None
    stage: str = "DOC_CLASSIFY"
    exception_type: str | None = None
    errors: list[str] | None = None


def normalize_document_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    key = text.lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "job_description": "job_description",
        "jobdescription": "job_description",
        "jd": "job_description",
        "job_posting": "job_description",
        "job_post": "job_description",
        "resume": "resume",
        "cv": "resume",
        "curriculum_vitae": "resume",
        "article": "article",
        "blog": "article",
        "tutorial": "article",
        "documentation": "documentation",
        "docs": "documentation",
        "readme": "documentation",
        "policy": "documentation",
        "rubric": "documentation",
        "roadmap": "documentation",
        "other": "other",
    }
    mapped = aliases.get(key)
    if mapped in ALLOWED_DOCUMENT_TYPES:
        return mapped
    return "other"


def normalize_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def clean_reject_reason(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    return text[:500]


def classify_error_message(
    purpose: Purpose,
    document_type: str | None,
    is_it_role: bool | None,
    reject_reason: str | None,
) -> str | None:
    """None nếu pass; message lỗi nếu reject."""
    allowed = _PASS_TYPES[purpose]
    if document_type not in allowed:
        return reject_reason or _DEFAULT_REJECT[purpose]
    if is_it_role is not True:
        return reject_reason or _DEFAULT_NOT_IT
    return None


def check_pass_rules(
    purpose: Purpose,
    document_type: str | None,
    is_it_role: bool | None,
    reject_reason: str | None = None,
) -> DocumentClassifyResult:
    """Gate deterministic sau khi đã có classify fields (không gọi LLM)."""
    err = classify_error_message(purpose, document_type, is_it_role, reject_reason)
    if err is None:
        return DocumentClassifyResult(
            success=True,
            document_type=document_type,
            is_it_role=True,
            reject_reason=None,
        )
    return DocumentClassifyResult(
        success=False,
        document_type=document_type,
        is_it_role=is_it_role,
        reject_reason=err,
        error="Tài liệu không hợp lệ cho hệ thống IT",
        detail=err,
        stage="DOC_CLASSIFY",
        exception_type="NotItDocument",
        errors=[err],
    )


class DocumentClassifyService:
    """Gọi LLM classify + optional L1 keyword trước."""

    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def classify(
        self,
        text: str,
        purpose: Purpose,
        *,
        run_l1: bool = True,
        max_chars: int = 8000,
    ) -> DocumentClassifyResult:
        excerpt = (text or "").strip()
        if not excerpt:
            msg = "Nội dung trống — không phân loại được."
            return DocumentClassifyResult(
                success=False,
                error="Nội dung trống",
                detail=msg,
                stage="DOC_CLASSIFY",
                exception_type="EmptyDocument",
                errors=[msg],
            )

        if run_l1:
            l1 = validate_it_domain(excerpt)
            if l1:
                return DocumentClassifyResult(
                    success=False,
                    error="Không thuộc lĩnh vực IT",
                    detail=l1,
                    stage="DOC_CLASSIFY",
                    exception_type="NotItDomainL1",
                    errors=[l1],
                )

        payload = excerpt[:max_chars]
        try:
            raw = self._call_llm(payload, purpose)
            parsed, err = extract_json_object(raw)
            if parsed is None and err and is_retryable_json_error(err):
                raw = self._call_llm(payload, purpose, fix_prompt=build_json_fix_prompt(raw))
                parsed, err = extract_json_object(raw)
            if parsed is None or not isinstance(parsed, dict):
                return DocumentClassifyResult(
                    success=False,
                    error="Không phân loại được tài liệu",
                    detail=err or "LLM không trả về JSON hợp lệ.",
                    stage="DOC_CLASSIFY",
                    exception_type="LlmJsonError",
                    errors=[err or "LLM không trả về JSON hợp lệ."],
                )

            document_type = normalize_document_type(
                parsed.get("documentType") or parsed.get("document_type")
            )
            is_it_role = normalize_bool(
                parsed.get("isItRole") if "isItRole" in parsed else parsed.get("is_it_role")
            )
            reject_reason = clean_reject_reason(
                parsed.get("rejectReason") or parsed.get("reject_reason")
            )
            return check_pass_rules(purpose, document_type, is_it_role, reject_reason)
        except Exception as ex:  # noqa: BLE001
            logger.exception("document classify failed purpose=%s", purpose)
            return DocumentClassifyResult(
                success=False,
                error="Phân loại tài liệu thất bại",
                detail=str(ex),
                stage="DOC_CLASSIFY",
                exception_type=type(ex).__name__,
                errors=[str(ex)],
            )

    def _call_llm(
        self, text: str, purpose: Purpose, *, fix_prompt: str | None = None
    ) -> str:
        hint = {
            "jd": "Ngữ cảnh: tin tuyển dụng (Job Description).",
            "cv": "Ngữ cảnh: CV/hồ sơ ứng viên.",
            "kb": "Ngữ cảnh: tài liệu knowledge base (policy/stack/rubric/roadmap).",
        }[purpose]
        user = f"{hint}\n\n--- NỘI DUNG ---\n{text}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        if fix_prompt:
            messages.append({"role": "user", "content": fix_prompt})
        response = self._client.chat.completions.create(
            model=self._settings.chat_model,
            messages=messages,
            temperature=0.1,
        )
        return (response.choices[0].message.content or "").strip()
