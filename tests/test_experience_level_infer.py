"""Tests cho chuẩn hóa experience_level từ output LLM."""
import pytest

from helpers.experience_level_infer import require_experience_level_from_llm


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
