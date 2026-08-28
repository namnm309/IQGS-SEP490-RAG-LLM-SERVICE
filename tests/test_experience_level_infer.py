"""Tests cho chuẩn hóa experience_level từ output LLM / HR confirm (SCRUM-417)."""
import pytest

from helpers.experience_level_infer import (
    require_experience_level_from_llm,
    require_hr_experience_level,
    resolve_experience_level,
)


def test_require_accepts_valid_llm_value() -> None:
    assert require_experience_level_from_llm("mid") == "mid"
    assert require_experience_level_from_llm("Senior") == "senior"


def test_require_rejects_missing() -> None:
    with pytest.raises(ValueError, match="experience_level"):
        require_experience_level_from_llm(None)
    with pytest.raises(ValueError, match="experience_level"):
        require_experience_level_from_llm("")


def test_require_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        require_experience_level_from_llm("expert")


def test_resolve_prefers_hr_confirmed() -> None:
    assert resolve_experience_level("mid", hr_confirmed="senior") == "senior"
    assert resolve_experience_level(None, hr_confirmed="Junior") == "junior"


def test_resolve_falls_back_to_llm_when_no_hr() -> None:
    assert resolve_experience_level("mid", hr_confirmed=None) == "mid"


def test_require_hr_rejects_missing() -> None:
    with pytest.raises(ValueError, match="HR phải xác nhận"):
        require_hr_experience_level(None)
    with pytest.raises(ValueError, match="HR phải xác nhận"):
        require_hr_experience_level("")


def test_require_hr_accepts_valid() -> None:
    assert require_hr_experience_level("Mid") == "mid"


def test_resolve_require_hr_ignores_llm() -> None:
    assert resolve_experience_level("mid", hr_confirmed="Senior", require_hr=True) == "senior"


def test_resolve_require_hr_raises_without_hr() -> None:
    with pytest.raises(ValueError, match="HR phải xác nhận"):
        resolve_experience_level("mid", hr_confirmed=None, require_hr=True)
