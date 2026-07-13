"""Chuẩn hóa experience_level từ output LLM — không suy heuristic từ JD."""
from models.internal_schemas import normalize_experience_level

__all__ = ["require_experience_level_from_llm"]


def require_experience_level_from_llm(raw_value: object) -> str:
    """Chỉ chấp nhận giá trị LLM trả về; thiếu/sai → ValueError."""
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        raise ValueError(
            "Plan thiếu experience_level — LLM phải suy luận từ JD, hrNote và RAG context."
        )
    return normalize_experience_level(str(raw_value))
