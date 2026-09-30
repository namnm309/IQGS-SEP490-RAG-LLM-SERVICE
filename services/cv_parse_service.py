"""Parse CV và trích kỹ năng bằng AI (SCRUM-300).

Hướng A: dùng một model duy nhất (settings.chat_model) cho cả tài liệu và ảnh.
- PDF/DOCX: trích text bằng DocumentParser rồi gửi text cho LLM qua OpenAI-compatible client.
- JPG/JPEG/PNG: gửi ảnh base64 tới Ollama native /api/chat (field images) — vì OpenAI-style
  image_url qua /v1 dễ khiến model Gemma "không thấy" ảnh.
"""
from __future__ import annotations

import base64
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

import httpx
from openai import OpenAI

from config.settings import Settings
from helpers.jd_validator import validate_it_domain
from models.internal_schemas import ParseCvResponse
from services.document_classify import (
    check_pass_rules,
    clean_reject_reason,
    normalize_bool,
    normalize_document_type,
)
from services.document_parser import DocumentParser
from services.json_output_parser import (
    build_json_fix_prompt,
    extract_json_object,
    is_retryable_json_error,
)
from services.rag_error_helpers import build_rag_error_detail

logger = logging.getLogger(__name__)

# Định dạng ảnh hỗ trợ (khớp allowlist Backend .NET)
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
DOCUMENT_EXTENSIONS = frozenset({".pdf", ".docx"})
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | DOCUMENT_EXTENSIONS

_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

CV_SYSTEM_PROMPT = """Bạn là trợ lý AI phân tích CV/hồ sơ ứng viên công nghệ (SCRUM-466).

## Nhiệm vụ
1) Phân loại: documentType + isItRole + rejectReason.
2) Nếu là CV IT: trích skills / summary / suggestedRole / yearsOfExperienceHint.

## Quy tắc classify (BẮT BUỘC)
1. documentType chỉ một trong: resume | article | documentation | other
   (resume = CV/hồ sơ cá nhân; không phải JD tuyển dụng).
2. isItRole = true chỉ khi hồ sơ CHÍNH thuộc IT/phần mềm.
3. Marketing/Sales/Kế toán/Luật dù có Excel → isItRole = false; documentType vẫn có thể là resume.
4. rejectReason: 1 câu tiếng Việt nếu không phải CV IT; null nếu pass (resume + isItRole=true).

## Quy tắc extract
5. CHỈ trả JSON hợp lệ, không markdown fence.
6. KHÔNG bịa kỹ năng không có trong CV.
7. skills: mảng chuỗi chuẩn hóa; [] nếu không có.
   - Tên ngắn, ASCII; dùng gạch nối "-" (không en-dash).
   - Acronym trong ngoặc OK: "Server-Driven UI (SDUI)".
   - Không emoji / ký tự đặc biệt lạ; không soft-skill / marketing.
8. summary tiếng Việt ngắn; chuỗi rỗng nếu thiếu thông tin.
9. TUYỆT ĐỐI KHÔNG suy luận level (intern/fresher/junior/middle/senior/lead).
10. suggestedRole chỉ gợi ý role — null nếu thiếu căn cứ.

## Schema JSON
{
  "documentType": "resume|article|documentation|other",
  "isItRole": true,
  "rejectReason": "string|null",
  "skills": ["string"],
  "summary": "string",
  "suggestedRole": "string|null",
  "yearsOfExperienceHint": number|null
}
"""

CV_USER_INSTRUCTION = (
    "Phân loại (documentType/isItRole) rồi trích kỹ năng từ CV sau. "
    "Không kết luận Fresher/Junior/Middle/Senior. Chỉ trả về JSON theo schema đã cho."
)


class CvParseService:
    def __init__(self, parser: DocumentParser, client: OpenAI, settings: Settings) -> None:
        self._parser = parser
        self._client = client
        self._settings = settings

    @property
    def _ollama_native_url(self) -> str:
        # Ollama native endpoint suy từ base URL hiện tại (bỏ hậu tố /v1) — hỗ trợ hot-reload SCRUM-378
        base = self._settings.ollama_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return f"{base}/api/chat"

    def parse_upload(
        self, file_bytes: bytes, file_name: str
    ) -> tuple[ParseCvResponse | None, dict | None, int]:
        """Trả (response, error_detail, status_code).

        - Thành công: (ParseCvResponse, None, 200)
        - Lỗi định dạng/đọc file: (None, error_detail, 422)
        - Lỗi AI/LLM: (None, error_detail, 502) — để BE throw và giữ nguyên TechStack cũ (AC-06)
        """
        if not file_name:
            return None, build_rag_error_detail(
                error="File không hợp lệ",
                detail="Thiếu tên file.",
                stage="CV_PARSE",
                exception_type="MissingFileName",
                errors=["Thiếu tên file."],
            ), 422

        if not file_bytes:
            return None, build_rag_error_detail(
                error="File CV trống",
                detail="Nội dung file rỗng.",
                stage="CV_PARSE",
                exception_type="EmptyDocument",
                errors=["Nội dung file rỗng."],
            ), 422

        ext = Path(file_name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return None, build_rag_error_detail(
                error="Định dạng file không hỗ trợ",
                detail=f"Chỉ hỗ trợ: {', '.join(sorted(SUPPORTED_EXTENSIONS))}. File: {file_name}",
                stage="CV_PARSE",
                exception_type="UnsupportedFileType",
                errors=[f"Định dạng file '{ext}' không hỗ trợ."],
            ), 422

        # Chuẩn bị caller gọi LLM tùy theo loại file
        try:
            if ext in IMAGE_EXTENSIONS:
                image_b64 = base64.b64encode(file_bytes).decode("ascii")
                caller: Callable[[str | None], str] = (
                    lambda fix: self._call_llm_vision(image_b64, fix_prompt=fix)
                )
            else:
                text = self._extract_document_text(file_bytes, file_name)
                # SCRUM-466 L1: keyword IT trước khi tốn token LLM
                l1_err = validate_it_domain(text)
                if l1_err:
                    return None, build_rag_error_detail(
                        error="CV không thuộc IT",
                        detail=l1_err,
                        stage="CV_CLASSIFY",
                        exception_type="NotItDomainL1",
                        errors=[l1_err],
                    ), 422
                caller = lambda fix: self._call_llm_text(text, fix_prompt=fix)
        except ValueError as exc:
            return None, build_rag_error_detail(
                error="Không đọc được file CV",
                detail=str(exc),
                stage="CV_PARSE",
                exception_type="EmptyDocument",
                errors=[str(exc)],
            ), 422
        except Exception as exc:  # noqa: BLE001 - lỗi đọc file bất ngờ
            logger.exception("parse-cv: đọc file thất bại")
            return None, build_rag_error_detail(
                error="Không đọc được file CV",
                detail=str(exc),
                stage="CV_PARSE",
                exception_type=type(exc).__name__,
                errors=[str(exc)],
            ), 422

        # Gọi LLM + parse JSON (retry 1 lần nếu JSON lỗi)
        try:
            raw = caller(None)
            parsed, err = extract_json_object(raw)
            if parsed is None and err and is_retryable_json_error(err):
                raw = caller(build_json_fix_prompt(raw))
                parsed, err = extract_json_object(raw)
        except Exception as exc:  # noqa: BLE001 - lỗi gọi model vision/LLM
            logger.exception("parse-cv: gọi AI thất bại")
            return None, build_rag_error_detail(
                error="Phân tích CV bằng AI thất bại",
                detail=str(exc),
                stage="CV_AI_ANALYSIS",
                exception_type=type(exc).__name__,
                errors=[str(exc)],
            ), 502

        if parsed is None or not isinstance(parsed, dict):
            return None, build_rag_error_detail(
                error="AI không trả về JSON hợp lệ",
                detail=err or "LLM không trả về JSON hợp lệ.",
                stage="CV_AI_ANALYSIS",
                exception_type="InvalidJsonOutput",
                errors=[err] if err else [],
            ), 502

        document_type = normalize_document_type(
            parsed.get("documentType") or parsed.get("document_type")
        )
        is_it_role = normalize_bool(
            parsed.get("isItRole") if "isItRole" in parsed else parsed.get("is_it_role")
        )
        reject_reason = clean_reject_reason(
            parsed.get("rejectReason") or parsed.get("reject_reason")
        )

        # SCRUM-466 L2: phải là resume IT
        classify = check_pass_rules("cv", document_type, is_it_role, reject_reason)
        if not classify.success:
            return None, build_rag_error_detail(
                error=classify.error or "CV không hợp lệ",
                detail=classify.detail,
                stage="CV_CLASSIFY",
                exception_type=classify.exception_type or "NotItDocument",
                errors=classify.errors or [],
            ), 422

        skills = self._normalize_skills(parsed.get("skills"))
        if not skills:
            return None, build_rag_error_detail(
                error="CV không có kỹ năng IT",
                detail="Không tìm thấy kỹ năng IT trong CV. Vui lòng tải lên CV kỹ thuật có tech stack rõ ràng.",
                stage="CV_CLASSIFY",
                exception_type="CvNoItSkills",
                errors=["Không tìm thấy kỹ năng IT trong CV."],
            ), 422

        summary = self._normalize_summary(parsed.get("summary"))
        suggested_role = self._normalize_optional_str(
            parsed.get("suggestedRole") or parsed.get("suggested_role")
        )
        years_hint = self._normalize_years(
            parsed.get("yearsOfExperienceHint") or parsed.get("years_of_experience_hint")
        )

        return ParseCvResponse(
            success=True,
            skills=skills,
            summary=summary,
            file_name=file_name,
            suggested_role=suggested_role,
            years_of_experience_hint=years_hint,
            document_type=document_type or "resume",
            is_it_role=True,
            reject_reason=None,
        ), None, 200

    def _extract_document_text(self, file_bytes: bytes, file_name: str) -> str:
        suffix = Path(file_name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)
        try:
            return self._parser.parse(tmp_path, file_name)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _call_llm_text(self, cv_text: str, *, fix_prompt: str | None = None) -> str:
        user_content = f"{CV_USER_INSTRUCTION}\n\n--- NỘI DUNG CV ---\n{cv_text}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": CV_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        if fix_prompt:
            messages.append({"role": "user", "content": fix_prompt})

        response = self._client.chat.completions.create(
            model=self._settings.chat_model,
            messages=messages,
            temperature=self._settings.temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def _call_llm_vision(self, image_b64: str, *, fix_prompt: str | None = None) -> str:
        # Ollama native /api/chat: ảnh đặt trong field images của message
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": CV_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": CV_USER_INSTRUCTION,
                "images": [image_b64],
            },
        ]
        if fix_prompt:
            messages.append({"role": "user", "content": fix_prompt})

        payload = {
            "model": self._settings.chat_model,
            "stream": False,
            "messages": messages,
            "options": {"temperature": self._settings.temperature},
        }
        response = httpx.post(
            self._ollama_native_url,
            json=payload,
            timeout=self._settings.request_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        return (data.get("message", {}).get("content") or "").strip()

    @staticmethod
    def _normalize_skills(raw: Any) -> list[str]:
        """SCRUM-493: sanitize skill CV — en-dash→-, giữ (), drop non-IT / char lạ."""
        if not isinstance(raw, list):
            return []
        non_it_exact = {
            "marketing",
            "marketting",
            "sales",
            "seo",
            "finance",
            "accounting",
            "accountant",
            "communication",
            "leadership",
            "teamwork",
            "excel",
            "hr",
        }
        result: list[str] = []
        seen: set[str] = set()
        for item in raw:
            text = str(item or "").strip()
            if not text:
                continue
            text = (
                text.replace("\u2013", "-")
                .replace("\u2014", "-")
                .replace("\u2212", "-")
                .replace("\u00ad", "-")
            )
            text = re.sub(r"\s+", " ", text).strip()
            text = "".join(
                c
                for c in text
                if c.isalnum() or c in " .#+/-()"
            )
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"\(\s*\)", "", text).strip()
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 40:
                text = text[:40].rstrip()
            if len(text) < 2 or not any(c.isalpha() for c in text):
                continue
            key = re.sub(r"[\s_\-]+", "", text.lower())
            if "marketing" in key or "marketting" in key:
                continue
            if key in non_it_exact:
                continue
            if key in seen:
                continue
            seen.add(key)
            result.append(text)
        return result

    @staticmethod
    def _normalize_summary(raw: Any) -> str | None:
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    @staticmethod
    def _normalize_optional_str(raw: Any) -> str | None:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        # Chặn giá trị level bị nhầm vào suggestedRole
        banned = {"fresher", "junior", "middle", "mid", "senior", "intern", "lead"}
        if text.lower() in banned:
            return None
        return text

    @staticmethod
    def _normalize_years(raw: Any) -> float | None:
        if raw is None or raw == "":
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        if value < 0 or value > 60:
            return None
        return value
