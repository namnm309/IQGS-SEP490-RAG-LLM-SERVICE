"""HG01: Domain (skill) / Why ask (rationale) / nguồn của câu phải khớp đúng slot outline."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from config.settings import Settings
from models.internal_schemas import (
    GeneratedQuestionItem,
    GenerateQuestionsFromPlanRequest,
    PlanCitationItem,
    QuestionGenerationPlan,
    QuestionTypeDistributionItem,
    RecommendedQuestionOutlineItem,
)
from services.question_generation_service import (
    QuestionGenerationService,
    _content_mismatches_slot,
    _reconcile_outline_types_to_distribution,
    _slot_mismatch_reasons,
    sanitize_outline_skill_goal,
)

OWNER_ID = "11111111-1111-1111-1111-111111111111"


def _settings() -> Settings:
    return Settings(
        DATABASE_URL="postgresql://postgres:postgres@localhost:5432/iqgs",
        BACKEND_INTERNAL_BASE_URL="http://localhost:5000",
        INTERNAL_API_KEY="test-secret",
        EMBEDDING_DIMENSION=768,
        DEBUG=False,
    )


def _slot(
    order: int, qtype: str, skill: str, goal: str, focus: str | None = None
) -> RecommendedQuestionOutlineItem:
    return RecommendedQuestionOutlineItem(
        order=order,
        type=qtype,
        difficulty="medium",
        skill=skill,
        focusArea=focus or skill,
        goal=goal,
    )


def _plan(
    outline: list[RecommendedQuestionOutlineItem],
    distribution: list[QuestionTypeDistributionItem] | None = None,
) -> QuestionGenerationPlan:
    return QuestionGenerationPlan(
        roleTitle="Fullstack Developer",
        experienceLevel="mid",
        totalQuestions=len(outline),
        difficulty="medium",
        skills=[o.skill for o in outline],
        questionTypeDistribution=distribution or [],
        recommendedQuestionOutline=outline,
    )


def _llm_reply(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    return MagicMock(choices=[choice])


def _service(*replies: str) -> QuestionGenerationService:
    client = MagicMock()
    client.chat.completions.create.side_effect = [_llm_reply(r) for r in replies]
    retrieval = MagicMock()
    retrieval.retrieve_for_job.return_value = ([], [])
    return QuestionGenerationService(retrieval=retrieval, client=client, settings=_settings())


def test_parse_keeps_slot_alignment_when_llm_returns_empty_question() -> None:
    """1 câu rỗng ở giữa không được làm các câu sau nhận order (→ Domain/Why ask) của slot trước."""
    outline = [
        _slot(1, "technical", "C#", "Check C# OOP fundamentals."),
        _slot(2, "technical", "ASP.NET Core", "Structure RESTful endpoints with Minimal APIs."),
        _slot(3, "technical", "Node.js", "Compare Node.js API implementation with .NET."),
        _slot(4, "technical", "React.js", "Build UI and manage state."),
    ]
    raw = json.dumps(
        {
            "questions": [
                {"order": 1, "question": "Interface vs abstract class in C#?", "skill": "C#"},
                {"order": 2, "question": "", "skill": "ASP.NET Core"},
                {"order": 3, "question": "Node.js vs ASP.NET Core for a REST API?", "skill": "Node.js"},
                {"order": 4, "question": "How do you manage state in React?", "skill": "React.js"},
            ]
        }
    )

    questions, _ = _service()._parse_questions_from_plan(raw, {}, _plan(outline))

    by_text = {q.question: q for q in questions}
    node_q = by_text["Node.js vs ASP.NET Core for a REST API?"]
    assert node_q.order == 3
    assert node_q.skill == "Node.js"
    assert node_q.rationale == "Compare Node.js API implementation with .NET."
    assert by_text["How do you manage state in React?"].skill == "React.js"


def test_llm_echo_of_another_slot_skill_is_a_mismatch() -> None:
    outline = [
        _slot(2, "technical", "Node.js", "Explain the Node.js event loop."),
        _slot(3, "technical", "React.js", "Explain React hooks."),
    ]
    by_order = {o.order: o for o in outline}
    q = GeneratedQuestionItem(
        question="How does the Node.js event loop schedule callbacks?",
        questionType="technical",
        difficulty="medium",
        order=3,
        skill="React.js",
        llm_skill_echo="Node.js",
    )

    reasons = _slot_mismatch_reasons(q, by_order[3], by_order)

    assert any(r.startswith("llm_skill_echo_mismatch") for r in reasons)


def test_llm_echo_same_skill_other_spelling_is_fine() -> None:
    slot = _slot(3, "technical", "React.js", "Explain React hooks.")
    q = GeneratedQuestionItem(
        question="Explain the useEffect hook in React.",
        questionType="technical",
        difficulty="medium",
        order=3,
        skill="React.js",
        llm_skill_echo="React",
    )

    assert _slot_mismatch_reasons(q, slot, {3: slot}) == []


def test_question_about_slot_focus_is_not_flagged() -> None:
    """Trước đây câu 'DI' ở slot C#/focus DI bị coi là lệch → regen 3 vòng vô ích."""
    slot = _slot(1, "technical", "C#", "Test DI", focus="DI")
    q = GeneratedQuestionItem(
        question="Câu hỏi 1 về DI?",
        questionType="technical",
        difficulty="medium",
        order=1,
        skill="C#",
    )

    assert _content_mismatches_slot(q, slot) == []


def test_reconcile_flip_to_behavioral_rewrites_whole_slot() -> None:
    outline = [
        _slot(i, "technical", f"Skill{i}", f"Goal about Skill{i}") for i in range(1, 8)
    ]
    outline.append(_slot(8, "behavioral", "Teamwork", "Handle conflicts"))
    outline.append(_slot(9, "situational", "Deadline", "Prioritize under pressure"))
    outline.append(_slot(10, "situational", "Outage", "Handle an outage"))
    for o in outline:
        o.citations = [
            PlanCitationItem(
                knowledge_base="hr",
                source_file="job-description",
                chunk_index=0,
                excerpt=f"JD:{o.skill}",
            )
        ]
    plan = _plan(
        outline,
        [
            QuestionTypeDistributionItem(type="technical", count=6),
            QuestionTypeDistributionItem(type="behavioral", count=2),
            QuestionTypeDistributionItem(type="situational", count=2),
        ],
    )

    changed = _reconcile_outline_types_to_distribution(
        plan, language="English", rewrite_slots=True
    )

    flipped = [o for o in plan.recommended_question_outline if o.relabeled]
    assert changed == 1
    assert len(flipped) == 1
    slot = flipped[0]
    assert slot.type == "behavioral"
    assert slot.skill == "Behavioral"
    assert "Skill" not in slot.goal
    assert slot.citations == []
    assert slot.planned_skill == "Behavioral"


def test_reconcile_without_rewrite_keeps_skill_for_coach_flow() -> None:
    outline = [_slot(i, "technical", f"Skill{i}", f"Goal{i}") for i in range(1, 4)]
    plan = _plan(
        outline,
        [
            QuestionTypeDistributionItem(type="technical", count=2),
            QuestionTypeDistributionItem(type="behavioral", count=1),
        ],
    )

    _reconcile_outline_types_to_distribution(plan)

    assert [o.skill for o in plan.recommended_question_outline] == ["Skill1", "Skill2", "Skill3"]


def test_prompt_does_not_leak_internal_markers() -> None:
    slot = _slot(1, "technical", "React.js", "Explain hooks.")
    slot.planned_skill = "React.js"
    slot.relabeled = True
    request = GenerateQuestionsFromPlanRequest(
        ownerId=OWNER_ID,
        jobDescription="Fullstack role",
        approvedPlan=_plan([slot]),
    )

    message = _service()._build_from_plan_user_message(request, [], [])

    assert "plannedSkill" not in message
    assert "relabeled" not in message
    assert "HG01" in message


def test_sanitize_goal_follows_output_language() -> None:
    outline = [_slot(1, "technical", "React.js", "Evaluate Node.js REST endpoints on .NET")]

    assert sanitize_outline_skill_goal(outline, "English") == 1
    assert outline[0].goal == "Assess hands-on React.js knowledge relevant to this role."


def test_outline_markers_round_trip_through_model() -> None:
    item = RecommendedQuestionOutlineItem.model_validate(
        {
            "order": 1,
            "type": "technical",
            "difficulty": "medium",
            "skill": "React.js",
            "goal": "g",
            "plannedSkill": "React.js",
            "relabeled": True,
        }
    )

    dumped = item.model_dump(by_alias=True)

    assert dumped["plannedSkill"] == "React.js"
    assert dumped["relabeled"] is True


def test_llm_skill_echo_is_not_serialized() -> None:
    q = GeneratedQuestionItem(
        question="Q?", questionType="technical", difficulty="medium", llm_skill_echo="Node.js"
    )

    assert "llm_skill_echo" not in q.model_dump()
    assert "llmSkillEcho" not in q.model_dump(by_alias=True)


def test_generate_from_plan_regens_slot_when_llm_answers_other_slot() -> None:
    outline = [
        _slot(1, "technical", "React.js", "Explain React hooks."),
        _slot(2, "technical", "Node.js", "Explain the Node.js event loop."),
    ]
    plan = _plan(outline, [QuestionTypeDistributionItem(type="technical", count=2)])
    first = json.dumps(
        {
            "questions": [
                {
                    "order": 1,
                    "question": "How does the Node.js event loop handle I/O callbacks?",
                    "skill": "Node.js",
                    "question_type": "technical",
                },
                {
                    "order": 2,
                    "question": "How does the Node.js event loop handle timers?",
                    "skill": "Node.js",
                    "question_type": "technical",
                },
            ]
        }
    )
    regen = json.dumps(
        {
            "questions": [
                {
                    "order": 1,
                    "question": "When would you use the useEffect hook in React?",
                    "skill": "React.js",
                    "question_type": "technical",
                }
            ]
        }
    )
    service = _service(first, regen)

    result = service.generate_from_plan(
        GenerateQuestionsFromPlanRequest(
            ownerId=OWNER_ID, jobDescription="Fullstack role", approvedPlan=plan
        )
    )

    q1 = next(q for q in result.questions if q.order == 1)
    assert "useEffect" in q1.question
    assert q1.skill == "React.js"
    assert q1.rationale == "Explain React hooks."
    assert not q1.needs_review
    assert service._client.chat.completions.create.call_count == 2
