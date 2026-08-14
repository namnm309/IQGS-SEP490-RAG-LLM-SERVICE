"""Chuẩn hóa ngôn ngữ đầu ra (Vietnamese | English) cho prompt LLM."""
from __future__ import annotations


def normalize_output_language(raw: str | None) -> str:
    """Trả về 'English' hoặc 'Vietnamese'."""
    if not raw or not str(raw).strip():
        return "Vietnamese"
    t = str(raw).strip().lower()
    if t in {"en", "english", "en-us", "en-gb"} or "eng" in t:
        return "English"
    if t in {"vi", "vn", "vietnamese"} or "viet" in t:
        return "Vietnamese"
    return "Vietnamese"


def language_instruction_block(raw: str | None) -> str:
    lang = normalize_output_language(raw)
    if lang == "English":
        return (
            "OUTPUT_LANGUAGE=English. "
            "Write ALL free-text fields (question, rationale, sample_answer, "
            "evaluation_criteria, summary, reason, goal, notes) in English."
        )
    return (
        "OUTPUT_LANGUAGE=Vietnamese. "
        "Write ALL free-text fields (question, rationale, sample_answer, "
        "evaluation_criteria, summary, reason, goal, notes) in Vietnamese (Tiếng Việt)."
    )


def free_text_language_label(raw: str | None) -> str:
    return "English" if normalize_output_language(raw) == "English" else "Vietnamese (Tiếng Việt)"
