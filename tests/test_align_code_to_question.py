"""Câu Git lý thuyết không được gắn snippet C# OrderService + stem code-heavy bắt buộc."""
from __future__ import annotations

from models.internal_schemas import GeneratedQuestionItem
from services.question_generation_service import (
    _align_code_to_question,
    _apply_outline_answer_method,
    _ensure_code_heavy_has_snippet,
    _is_generic_fallback_snippet,
)


def test_align_strips_orderservice_from_git_theory_question() -> None:
    q = GeneratedQuestionItem(
        question="Trong quy trình Git, Pull Request đóng vai trò gì?",
        question_type="technical",
        difficulty="easy",
        skill="Git",
        focus_area="Git",
        rationale="Kiểm tra hiểu PR",
        answer_method="Code",
        code_template_type="CODE_COMPLETION",
        code_snippet=(
            "public class OrderService\n"
            "{\n"
            "    private readonly IOrderRepository _orders;\n"
            "    public async Task<decimal> GetCustomerOrderTotalAsync(int customerId)\n"
            "    { throw new NotImplementedException(); }\n"
            "}"
        ),
    )
    assert _is_generic_fallback_snippet(q.code_snippet)
    _align_code_to_question(q, "Mixed")
    assert q.code_snippet is None
    assert q.code_template_type is None
    assert q.answer_method == "Text"


def test_outline_text_keeps_bug_detection_stem() -> None:
    """Answer=Text chỉ là cách trả lời — giữ code_snippet đề BUG_DETECTION."""
    stem = (
        "[HttpPut(\"{id}\")]\n"
        "public async Task<IActionResult> Update(int id, User user)\n"
        "{\n"
        "    _db.Users.Update(user);\n"
        "    await _db.SaveChangesAsync();\n"
        "    return Ok(user);\n"
        "}"
    )
    q = GeneratedQuestionItem(
        question="Hãy phân tích đoạn code sau. Tìm lỗi logic liên quan over-posting.",
        question_type="technical",
        difficulty="hard",
        skill="RESTful APIs",
        answer_method="Code",
        code_template_type="BUG_DETECTION",
        code_snippet=stem,
    )
    _apply_outline_answer_method(q, "Text")
    assert q.answer_method == "Text"
    assert q.code_template_type == "BUG_DETECTION"
    assert q.code_snippet == stem


def test_outline_text_clears_theory_without_stem() -> None:
    q = GeneratedQuestionItem(
        question="Giải thích REST là gì?",
        question_type="technical",
        difficulty="easy",
        answer_method="Code",
        code_template_type=None,
        code_snippet=None,
    )
    _apply_outline_answer_method(q, "Text")
    assert q.answer_method == "Text"
    assert q.code_snippet is None
    assert q.code_template_type is None


def test_ensure_injects_default_when_bug_detection_missing_stem() -> None:
    q = GeneratedQuestionItem(
        question="Tìm lỗi logic trong đoạn code sau.",
        question_type="technical",
        difficulty="medium",
        code_template_type="BUG_DETECTION",
        code_snippet=None,
        answer_method="Text",
    )
    _ensure_code_heavy_has_snippet(q)
    assert q.code_snippet
    assert "SumPositive" in q.code_snippet or "for" in q.code_snippet


def test_align_mixed_keeps_bug_detection_with_stem() -> None:
    stem = (
        "public int Divide(int a, int b)\n"
        "{\n"
        "    return a / b;\n"
        "}"
    )
    q = GeneratedQuestionItem(
        question="Hãy phân tích đoạn code sau và tìm lỗi logic chia cho zero.",
        question_type="technical",
        difficulty="medium",
        skill="C#",
        code_template_type="BUG_DETECTION",
        code_snippet=stem,
        answer_method="Text",
    )
    _align_code_to_question(q, "Mixed")
    _ensure_code_heavy_has_snippet(q)
    assert q.code_template_type == "BUG_DETECTION"
    assert q.code_snippet == stem
    assert q.answer_method == "Text"
