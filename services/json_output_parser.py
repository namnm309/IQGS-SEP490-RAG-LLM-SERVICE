"""Parse JSON từ LLM output — strip markdown fences, extract object, retry prompt."""
from __future__ import annotations

import json
import re
from typing import Any

JSON_FIX_USER_PROMPT = (
    "Lần trước bạn trả về JSON không hợp lệ. "
    "Chỉ sửa và trả về JSON hợp lệ theo schema, không markdown, không giải thích ngoài JSON.\n"
    "Output lỗi trước:\n{invalid_output}"
)


def strip_markdown_fences(text: str) -> str:
    """Loại bỏ ```json ... ``` nếu model bọc output trong markdown."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    return re.sub(r"\s*```$", "", stripped).strip()


def extract_json_object(text: str) -> tuple[Any | None, str | None]:
    """
    Parse JSON từ raw LLM output.
    Trả (parsed_data, None) nếu thành công, (None, error_message) nếu thất bại.
    """
    cleaned = strip_markdown_fences(text)

    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None, "LLM không trả về JSON hợp lệ."

    try:
        return json.loads(match.group()), None
    except json.JSONDecodeError as exc:
        return None, f"Không parse được JSON: {exc}"


def build_json_fix_prompt(invalid_output: str, *, max_len: int = 2000) -> str:
    """Tạo user prompt cho lần retry khi JSON invalid."""
    return JSON_FIX_USER_PROMPT.format(invalid_output=invalid_output[:max_len])


def is_retryable_json_error(error: str) -> bool:
    """Lỗi parse JSON thuần — có thể retry một lần."""
    json_error_phrases = (
        "LLM không trả về JSON hợp lệ",
        "Không parse được JSON",
        "JSON thiếu object",
    )
    return any(phrase in error for phrase in json_error_phrases)


PLAN_VALIDATION_FIX_USER_PROMPT = (
    "JSON parse được nhưng thiếu hoặc sai field bắt buộc. "
    "Sửa và trả về JSON đầy đủ theo schema — TẤT CẢ field phải do bạn suy luận từ "
    "jobDescription, hrNote, skills HR gửi và context RAG [HỆ THỐNG]/[HR]. "
    "Không bỏ trống role_title, summary, difficulty, level, experience_level, skills.\n"
    "Lỗi validation:\n{validation_error}\n"
    "Output trước:\n{invalid_output}"
)


def build_plan_validation_fix_prompt(validation_error: str, invalid_output: str, *, max_len: int = 2000) -> str:
    return PLAN_VALIDATION_FIX_USER_PROMPT.format(
        validation_error=validation_error,
        invalid_output=invalid_output[:max_len],
    )


def is_retryable_plan_error(error: str) -> bool:
    """Retry khi JSON malformed hoặc LLM thiếu/sai field bắt buộc."""
    if is_retryable_json_error(error):
        return True
    retryable_phrases = (
        "Plan thiếu",
        "không khớp yêu cầu",
        "experience_level",
        "experienceLevel",
        "roleTitle",
        "role_title",
        "level",
        "summary",
        "skills",
    )
    return any(phrase in error for phrase in retryable_phrases)
