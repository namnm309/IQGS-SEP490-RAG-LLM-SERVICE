"""Parse CV và trích kỹ năng bằng AI (SCRUM-300).

Hướng A: dùng một model duy nhất (settings.chat_model) cho cả tài liệu và ảnh.
- PDF/DOCX: trích text bằng DocumentParser rồi gửi text cho LLM qua OpenAI-compatible client.
- JPG/JPEG/PNG: gửi ảnh base64 tới Ollama native /api/chat (field images) — vì OpenAI-style
  image_url qua /v1 dễ khiến model Gemma "không thấy" ảnh.
"""
from __future__ import annotations

import base64
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

import httpx
from openai import OpenAI

from config.settings import Settings
from models.internal_schemas import ParseCvResponse
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

CV_SYSTEM_PROMPT = """Bạn là trợ lý AI phân tích CV/hồ sơ ứng viên công nghệ.

## Nhiệm vụ
Đọc nội dung CV (văn bản hoặc ảnh) và trích xuất:
1. skills: danh sách kỹ năng công nghệ (ngôn ngữ, framework, database, công cụ, nền tảng...).
2. summary: tóm tắt ngắn 1-3 câu về ứng viên (vai trò, kinh nghiệm nổi bật).

## Quy tắc bắt buộc
1. CHỈ trả về JSON hợp lệ, không markdown fence, không text ngoài JSON.
2. KHÔNG bịa kỹ năng không xuất hiện trong CV.
3. skills là mảng chuỗi, mỗi phần tử là tên kỹ năng chuẩn hóa (ví dụ "ASP.NET Core", "PostgreSQL").
4. Nếu không tìm thấy kỹ năng nào, trả skills = [].
5. summary viết bằng tiếng Việt, ngắn gọn; nếu không đủ thông tin để tóm tắt thì để chuỗi rỗng.

## Schema JSON
{
  "skills": ["string"],
  "summary": "string"
}
"""

CV_USER_INSTRUCTION = (
    "Trích kỹ năng và tóm tắt từ CV sau. Chỉ trả về JSON theo schema đã cho."
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

        skills = self._normalize_skills(parsed.get("skills"))
        summary = self._normalize_summary(parsed.get("summary"))

        return ParseCvResponse(
            success=True,
            skills=skills,
            summary=summary,
            file_name=file_name,
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
        if not isinstance(raw, list):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for item in raw:
            text = str(item or "").strip()
            if not text:
                continue
            key = text.lower()
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
