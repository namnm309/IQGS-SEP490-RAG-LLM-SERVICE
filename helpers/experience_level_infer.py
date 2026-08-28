"""Chuẩn hóa experience_level — SCRUM-417: Studio chỉ lấy từ HR, không suy JD."""
from models.internal_schemas import normalize_experience_level

__all__ = [
    "require_experience_level_from_llm",
    "resolve_experience_level",
    "require_hr_experience_level",
]


def require_hr_experience_level(hr_confirmed: str | None) -> str:
    """HR Studio: bắt buộc có experienceLevel trên request — thiếu → lỗi."""
    if hr_confirmed is None or (isinstance(hr_confirmed, str) and not hr_confirmed.strip()):
        raise ValueError(
            "Plan thiếu experience_level — HR phải xác nhận cấp độ (Intern|Junior|Mid|Senior|Lead), "
            "không suy từ JD."
        )
    return normalize_experience_level(str(hr_confirmed))


def require_experience_level_from_llm(raw_value: object) -> str:
    """Legacy (non-Studio): chấp nhận giá trị LLM trả về."""
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        raise ValueError(
            "Plan thiếu experience_level — LLM phải trả về intern|junior|mid|senior|lead."
        )
    return normalize_experience_level(str(raw_value))


def resolve_experience_level(
    llm_raw: object,
    *,
    hr_confirmed: str | None = None,
    require_hr: bool = False,
) -> str:
    """require_hr=True (Studio): chỉ HR; False (legacy): fallback LLM output."""
    if require_hr:
        return require_hr_experience_level(hr_confirmed)
    if hr_confirmed is not None and str(hr_confirmed).strip():
        return normalize_experience_level(str(hr_confirmed))
    return require_experience_level_from_llm(llm_raw)
