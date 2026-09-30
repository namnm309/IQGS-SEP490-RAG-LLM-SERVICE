"""SCRUM-495: distribution HR + content↔slot (regen, không che lệch bằng nhãn)."""
from __future__ import annotations

from collections import Counter

from models.internal_schemas import (
    GeneratedQuestionItem,
    QuestionGenerationPlan,
    QuestionTypeDistributionItem,
    RecommendedQuestionOutlineItem,
)
from services.question_generation_service import (
    QuestionGenerationService,
    _content_mismatch_reasons,
    _content_mismatches_slot,
    _expected_type_counts,
    _reassign_mismatched_contents,
    _reconcile_outline_types_to_distribution,
    _safe_normalize_question_type,
    _validate_questions_against_plan,
    sanitize_outline_skill_goal,
)


def _outline_item(
    order: int,
    qtype: str,
    skill: str,
    goal: str,
) -> RecommendedQuestionOutlineItem:
    return RecommendedQuestionOutlineItem(
        order=order,
        type=qtype,
        difficulty="medium",
        skill=skill,
        focusArea=skill,
        goal=goal,
    )


def _plan_dist_6_2_2(outline: list[RecommendedQuestionOutlineItem]) -> QuestionGenerationPlan:
    return QuestionGenerationPlan(
        roleTitle="Fullstack Developer",
        experienceLevel="mid",
        totalQuestions=10,
        difficulty="medium",
        questionTypeDistribution=[
            QuestionTypeDistributionItem(type="technical", count=6),
            QuestionTypeDistributionItem(type="behavioral", count=2),
            QuestionTypeDistributionItem(type="situational", count=2),
        ],
        recommendedQuestionOutline=outline,
    )


def test_safe_normalize_aliases() -> None:
    assert _safe_normalize_question_type("Behavioural") == "behavioral"
    assert _safe_normalize_question_type("culture-fit") == "situational"


def test_expected_counts_prefer_distribution_not_outline() -> None:
    outline = [_outline_item(i, "technical", f"S{i}", f"G{i}") for i in range(1, 8)]
    outline.append(_outline_item(8, "behavioral", "Team", "Conflict"))
    outline.append(_outline_item(9, "situational", "Dead", "Priority"))
    outline.append(_outline_item(10, "situational", "Inc", "Outage"))
    plan = _plan_dist_6_2_2(outline)
    counts = _expected_type_counts(plan)
    assert counts["technical"] == 6
    assert counts["behavioral"] == 2
    assert counts["situational"] == 2


def test_reconcile_outline_7_1_2_to_6_2_2() -> None:
    outline = [_outline_item(i, "technical", f"S{i}", f"G{i}") for i in range(1, 8)]
    outline.append(_outline_item(8, "behavioral", "Team", "Conflict"))
    outline.append(_outline_item(9, "situational", "Dead", "Priority"))
    outline.append(_outline_item(10, "situational", "Inc", "Outage"))
    plan = _plan_dist_6_2_2(outline)
    changed = _reconcile_outline_types_to_distribution(plan)
    assert changed >= 1
    after = Counter(
        _safe_normalize_question_type(o.type) for o in plan.recommended_question_outline
    )
    assert after["technical"] == 6
    assert after["behavioral"] == 2
    assert after["situational"] == 2


def test_lock_after_reconcile_matches_distribution() -> None:
    outline = [_outline_item(i, "technical", f"S{i}", f"G{i}") for i in range(1, 8)]
    outline.append(_outline_item(8, "behavioral", "Team", "Xử lý xung đột trong team"))
    outline.append(_outline_item(9, "situational", "Dead", "Khi deadline gấp"))
    outline.append(_outline_item(10, "situational", "Inc", "Khi outage"))
    plan = _plan_dist_6_2_2(outline)
    _reconcile_outline_types_to_distribution(plan)
    questions = [
        GeneratedQuestionItem(
            question=f"Q{i} technical docker typescript react postgres",
            questionType="technical",
            difficulty="medium",
            order=i,
            skill=f"S{i}",
        )
        for i in range(1, 11)
    ]
    QuestionGenerationService._lock_questions_to_outline(questions, plan)
    err = _validate_questions_against_plan(questions, plan, enforce_distribution=True)
    assert err is None


def test_content_mismatch_postgres_vs_nextjs_slot() -> None:
    outline = _outline_item(5, "technical", "Next.js", "Routing App Router SSR SSG")
    q = GeneratedQuestionItem(
        question="Làm sao tối ưu truy vấn PostgreSQL chậm với EXPLAIN ANALYZE?",
        questionType="technical",
        difficulty="medium",
        rationale="Routing App Router SSR SSG",
        order=5,
        skill="Next.js",
    )
    reasons = _content_mismatches_slot(q, outline)
    assert any("content_mismatch_skill" in r for r in reasons)


def test_reassign_swaps_postgres_and_nextjs() -> None:
    o3 = _outline_item(3, "technical", "PostgreSQL", "Tối ưu query SQL")
    o5 = _outline_item(5, "technical", "Next.js", "Routing App Router SSR")
    outline_by_order = {3: o3, 5: o5}
    q3 = GeneratedQuestionItem(
        question="Giải thích SSR và SSG trong Next.js khi render trang?",
        questionType="technical",
        difficulty="medium",
        order=3,
        skill="PostgreSQL",
        rationale="Tối ưu query SQL",
    )
    q5 = GeneratedQuestionItem(
        question="Làm sao tối ưu truy vấn PostgreSQL với index và EXPLAIN?",
        questionType="technical",
        difficulty="medium",
        order=5,
        skill="Next.js",
        rationale="Routing App Router SSR",
    )
    assert _content_mismatch_reasons(q3, o3)
    assert _content_mismatch_reasons(q5, o5)
    swaps = _reassign_mismatched_contents([q3, q5], outline_by_order)
    assert swaps >= 1
    assert _content_mismatch_reasons(q3, o3) == []
    assert _content_mismatch_reasons(q5, o5) == []


def test_hr30_react_slot_rest_content_mismatches_does_not_rewrite_skill() -> None:
    """HR30 #3: slot React + goal Node/.NET + câu REST (có thể nhắc React) → mismatch; không đổi Domain."""
    outline = _outline_item(
        3,
        "technical",
        "React.js",
        "Đánh giá kiến thức Node.js và ASP.NET Core khi thiết kế REST API",
    )
    q = GeneratedQuestionItem(
        question=(
            "Trong React app, thiết kế REST API cho /orders: HTTP method nào phù hợp "
            "để update một phần resource (PATCH vs PUT)?"
        ),
        questionType="technical",
        difficulty="medium",
        order=3,
        skill="React.js",
        rationale="Đánh giá kiến thức Node.js và ASP.NET Core khi thiết kế REST API",
    )
    reasons = _content_mismatches_slot(q, outline)
    assert any("content_mismatch_skill" in r for r in reasons)

    # Lock giữ skill React — không rewrite thành REST API
    plan = _plan_dist_6_2_2([outline])
    QuestionGenerationService._lock_questions_to_outline([q], plan)
    assert q.skill == "React.js"
    assert "Node" in (q.rationale or "") or "ASP.NET" in (q.rationale or "")


def test_next_ssr_vs_hooks_goal_mismatches_slot() -> None:
    """SSR/SSG content vs goal hooks → mismatch (regen), không vá Why ask."""
    outline = _outline_item(
        4,
        "technical",
        "Next.js",
        "Check basic knowledge of React hooks and component-based architecture.",
    )
    q = GeneratedQuestionItem(
        question=(
            "In Next.js, what is the primary difference between Server-Side Rendering (SSR) "
            "and Static Site Generation (SSG)?"
        ),
        questionType="technical",
        difficulty="easy",
        order=4,
        skill="Next.js",
        rationale="Check basic knowledge of React hooks and component-based architecture.",
    )
    reasons = _content_mismatches_slot(q, outline)
    assert any("content_mismatch_rationale_goal" in r for r in reasons)
    # Metadata không bị rewrite bởi stub align
    from services.question_generation_service import _align_why_ask_and_skill_to_content

    before_skill, before_r = q.skill, q.rationale
    assert _align_why_ask_and_skill_to_content([q], {4: outline}) == 0
    assert q.skill == before_skill
    assert q.rationale == before_r


def test_sanitize_outline_goal_leaked_backend_on_react_skill() -> None:
    outline = [
        _outline_item(
            1,
            "technical",
            "React.js",
            "Đánh giá Node.js event loop và REST trên .NET",
        )
    ]
    changed = sanitize_outline_skill_goal(outline)
    assert changed == 1
    assert outline[0].goal == "Đánh giá năng lực React.js"
    assert "Node" not in outline[0].goal


def test_lock_overwrites_llm_skill() -> None:
    plan = _plan_dist_6_2_2(
        [_outline_item(1, "technical", "Node.js", "So sánh Node và ASP.NET")]
    )
    questions = [
        GeneratedQuestionItem(
            question="So sánh Node.js và ASP.NET Core về event loop?",
            questionType="behavioral",
            difficulty="medium",
            rationale="LLM viết lung tung",
            order=1,
            skill="React.js",
        )
    ]
    QuestionGenerationService._lock_questions_to_outline(questions, plan)
    assert questions[0].question_type == "technical"
    assert questions[0].skill == "Node.js"
    assert questions[0].rationale == "So sánh Node và ASP.NET"
