"""Phân tích JD bằng LLM — job profile structured (SCRUM-416 + SCRUM-432).

Quy tắc: KHÔNG bịa. Không chắc → null / []. Không default "Software Engineer".
SCRUM-432: classify documentType + isItRole trước khi coi là tin tuyển dụng IT.
"""
from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from config.settings import Settings
from models.internal_schemas import AnalyzeJdRequest, AnalyzeJdResponse
from services.json_output_parser import (
    build_json_fix_prompt,
    extract_json_object,
    is_retryable_json_error,
)

logger = logging.getLogger(__name__)

ALLOWED_DOCUMENT_TYPES = frozenset(
    {"job_description", "resume", "article", "documentation", "other"}
)

ANALYZE_JD_SYSTEM_PROMPT = """Bạn là trợ lý HR phân tích văn bản tuyển dụng.

## Nhiệm vụ
1) Phân loại văn bản (documentType + isItRole).
2) Nếu là tin tuyển dụng IT: trích xuất metadata. CHỈ lấy thông tin có trong văn bản.

## Quy tắc classify (BẮT BUỘC)
1. documentType chỉ một trong:
   - job_description: đang TUYỂN NGƯỜI — có vị trí + việc phải làm + yêu cầu.
   - resume: CV / hồ sơ ứng viên (kinh nghiệm cá nhân, "tôi đã làm…").
   - article: blog / tutorial / bài viết kỹ thuật (dạy / giải thích, không tuyển).
   - documentation: README / API docs / tài liệu sản phẩm.
   - other: không thuộc các loại trên.
2. isItRole = true chỉ khi CÔNG VIỆC CHÍNH thuộc IT/phần mềm
   (dev, QA, DevOps, data, BA/PM kỹ thuật, security, cloud…).
3. Tutorial React, CV Senior .NET, README NestJS → KHÔNG phải job_description.
4. JD Marketing/Sales/Kế toán dù có chữ Excel/SQL → isItRole = false.
5. rejectReason: 1 câu tiếng Việt giải thích vì sao reject; null nếu pass
   (documentType=job_description VÀ isItRole=true).

## Quy tắc extract (khi pass classify)
6. CHỈ trả JSON hợp lệ, không markdown fence, không text ngoài JSON.
7. KHÔNG bịa. Không đủ căn cứ → null hoặc [].
8. KHÔNG default "Software Engineer" / "Mid" nếu văn bản không nói rõ.
9. jobTitle / position: chức danh tuyển. null nếu không rõ.
10. detectedRole: vai trò chuyên môn ngắn. null nếu không rõ.
11. experienceLevel / detectedSeniority: intern|junior|mid|senior|lead — hoặc null.
12. detectedLanguage: vietnamese | english | null.
13. skills: tối đa 12; [] nếu không có.
14. responsibilities: tối đa 10 bullet; [] nếu không có.
15. summary: 1-3 câu hoặc null.

## Schema JSON
{
  "documentType": "job_description|resume|article|documentation|other",
  "isItRole": true,
  "rejectReason": "string | null",
  "jobTitle": "string | null",
  "position": "string | null",
  "detectedRole": "string | null",
  "experienceLevel": "intern|junior|mid|senior|lead|null",
  "detectedSeniority": "Intern|Junior|Mid|Senior|Lead|null",
  "detectedLanguage": "vietnamese|english|null",
  "skills": ["string"],
  "responsibilities": ["string"],
  "summary": "string | null"
}
"""


class JdAnalyzeService:
    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    def analyze(self, request: AnalyzeJdRequest) -> AnalyzeJdResponse:
        text = (request.job_description or "").strip()
        if not text:
            return AnalyzeJdResponse(
                success=False,
                error="JD trống",
                detail="Nội dung JD không được rỗng.",
                stage="JD_ANALYZE",
                exception_type="EmptyJd",
                errors=["Nội dung JD không được rỗng."],
            )

        try:
            payload = (
                "Phân loại rồi trích metadata từ văn bản sau. Chỉ trả JSON theo schema. "
                "Không chắc thì null/[].\n\n"
                f"--- TEXT ---\n{text[:12000]}"
            )
            raw = self._call_llm(payload)
            parsed, err = extract_json_object(raw)
            if parsed is None and err and is_retryable_json_error(err):
                raw = self._call_llm(payload, fix_prompt=build_json_fix_prompt(raw))
                parsed, err = extract_json_object(raw)

            if parsed is None:
                return AnalyzeJdResponse(
                    success=False,
                    error="Không phân tích được JD",
                    detail=err or "LLM không trả về JSON hợp lệ.",
                    stage="JD_ANALYZE",
                    exception_type="LlmJsonError",
                    errors=[err or "LLM không trả về JSON hợp lệ."],
                )

            document_type = self._normalize_document_type(
                parsed.get("documentType") or parsed.get("document_type")
            )
            is_it_role = self._normalize_bool(
                parsed.get("isItRole") if "isItRole" in parsed else parsed.get("is_it_role")
            )
            reject_reason = self._clean_reject_reason(
                parsed.get("rejectReason") or parsed.get("reject_reason")
            )

            # SCRUM-432: fail classify → không trả metadata thành công
            classify_error = self._classify_error(document_type, is_it_role, reject_reason)
            if classify_error is not None:
                return AnalyzeJdResponse(
                    success=False,
                    document_type=document_type,
                    is_it_role=is_it_role,
                    reject_reason=classify_error,
                    error="Không phải tin tuyển dụng IT hợp lệ",
                    detail=classify_error,
                    stage="JD_CLASSIFY",
                    exception_type="NotItJobPosting",
                    errors=[classify_error],
                )

            job_title = self._clean_str(
                parsed.get("jobTitle") or parsed.get("job_title") or parsed.get("position")
            )
            seniority_display = self._normalize_seniority_display(
                parsed.get("detectedSeniority")
                or parsed.get("detected_seniority")
                or parsed.get("experienceLevel")
                or parsed.get("experience_level")
            )
            exp_level = self._normalize_experience_level(
                parsed.get("experienceLevel")
                or parsed.get("experience_level")
                or seniority_display
            )

            return AnalyzeJdResponse(
                success=True,
                document_type=document_type or "job_description",
                is_it_role=True,
                reject_reason=None,
                position=job_title,
                job_title=job_title,
                detected_role=self._clean_str(parsed.get("detectedRole") or parsed.get("detected_role")),
                detected_seniority=seniority_display,
                experience_level=exp_level,
                detected_language=self._normalize_language(
                    parsed.get("detectedLanguage") or parsed.get("detected_language")
                ),
                skills=self._clean_skills(parsed.get("skills")),
                responsibilities=self._clean_responsibilities(parsed.get("responsibilities")),
                summary=self._clean_summary(parsed.get("summary")),
            )
        except Exception as ex:
            logger.exception("analyze-jd failed")
            return AnalyzeJdResponse(
                success=False,
                error="Phân tích JD thất bại",
                detail=str(ex),
                stage="JD_ANALYZE",
                exception_type=type(ex).__name__,
                errors=[str(ex)],
            )

    def _call_llm(self, user_payload: str, *, fix_prompt: str | None = None) -> str:
        messages = [
            {"role": "system", "content": ANALYZE_JD_SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ]
        if fix_prompt:
            messages.append({"role": "user", "content": fix_prompt})
        response = self._client.chat.completions.create(
            model=self._settings.chat_model,
            messages=messages,
            temperature=0.1,
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _classify_error(
        document_type: str | None,
        is_it_role: bool | None,
        reject_reason: str | None,
    ) -> str | None:
        """Trả message lỗi classify; None nếu pass."""
        if document_type != "job_description":
            return reject_reason or (
                "Đây không phải tin tuyển dụng (Job Description). "
                "Vui lòng dán/upload JD đang tuyển vị trí IT/phần mềm."
            )
        if is_it_role is not True:
            return reject_reason or (
                "Vị trí không thuộc lĩnh vực IT/phần mềm. "
                "Hệ thống chỉ nhận tin tuyển dụng kỹ thuật."
            )
        return None

    @staticmethod
    def _normalize_document_type(value: Any) -> str | None:
        text = JdAnalyzeService._clean_str(value)
        if not text:
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
            "other": "other",
        }
        mapped = aliases.get(key)
        if mapped in ALLOWED_DOCUMENT_TYPES:
            return mapped
        return "other"

    @staticmethod
    def _normalize_bool(value: Any) -> bool | None:
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

    @staticmethod
    def _clean_reject_reason(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"null", "none"}:
            return None
        return text[:500]

    @staticmethod
    def _clean_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"null", "none", "n/a", "unknown", "không xác định"}:
            return None
        return text[:150]

    @staticmethod
    def _clean_summary(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() in {"null", "none"}:
            return None
        return text[:2000]

    @staticmethod
    def _normalize_seniority_display(value: Any) -> str | None:
        text = JdAnalyzeService._clean_str(value)
        if not text:
            return None
        key = text.lower().replace(" ", "").replace("-", "")
        mapping = {
            "intern": "Intern",
            "internship": "Intern",
            "fresher": "Intern",
            "junior": "Junior",
            "entry": "Junior",
            "entrylevel": "Junior",
            "mid": "Mid",
            "middle": "Mid",
            "midlevel": "Mid",
            "senior": "Senior",
            "sr": "Senior",
            "lead": "Lead",
            "principal": "Lead",
            "staff": "Lead",
        }
        return mapping.get(key)

    @staticmethod
    def _normalize_experience_level(value: Any) -> str | None:
        text = JdAnalyzeService._clean_str(value)
        if not text:
            return None
        key = text.lower().replace(" ", "").replace("-", "")
        mapping = {
            "intern": "intern",
            "internship": "intern",
            "fresher": "intern",
            "junior": "junior",
            "entry": "junior",
            "entrylevel": "junior",
            "mid": "mid",
            "middle": "mid",
            "midlevel": "mid",
            "senior": "senior",
            "sr": "senior",
            "lead": "lead",
            "principal": "lead",
            "staff": "lead",
        }
        return mapping.get(key)

    @staticmethod
    def _normalize_language(value: Any) -> str | None:
        text = JdAnalyzeService._clean_str(value)
        if not text:
            return None
        lower = text.lower()
        if "viet" in lower or lower in {"vi", "vn"}:
            return "vietnamese"
        if "eng" in lower or lower == "en":
            return "english"
        return None

    @staticmethod
    def _clean_skills(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            s = JdAnalyzeService._clean_str(item)
            if s and s not in out:
                out.append(s)
            if len(out) >= 12:
                break
        return out

    @staticmethod
    def _clean_responsibilities(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            s = JdAnalyzeService._clean_str(item)
            if s and len(s) <= 500 and s not in out:
                out.append(s)
            if len(out) >= 10:
                break
        return out
