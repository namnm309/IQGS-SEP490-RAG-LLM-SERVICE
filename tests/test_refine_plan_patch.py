"""SCRUM-420: PlanPatch schema + merge helper."""
from __future__ import annotations

import pytest

from models.internal_schemas import PlanPatch, RefinePlanRequest, SkillCoverageItem
from services.plan_refine_service import _merge_baseline_patch, _normalize_patch_dict


def test_plan_patch_has_changes():
    patch = PlanPatch(replace_skills=["Git"], instruction_applied="Git only")
    assert patch.has_changes() is True

    empty = PlanPatch()
    assert empty.has_changes() is False


def test_normalize_patch_dict_snake_case():
    raw = {
        "replace_coverage": [
            {
                "skill": "Git",
                "question_count": 5,
                "focus_areas": ["syntax"],
                "source_files": ["git.pdf"],
            }
        ],
        "update_summary": "Git focus",
    }
    normalized = _normalize_patch_dict(raw)
    assert "replaceCoverage" in normalized
    item = SkillCoverageItem.model_validate(normalized["replaceCoverage"][0])
    assert item.skill == "Git"
    assert item.question_count == 5


def test_merge_baseline_git_patch():
    baseline = {
        "roleTitle": "Dev",
        "summary": "Old summary",
        "totalQuestions": 10,
        "skills": ["C#", "SQL"],
        "coverage": [
            {"skill": "C#", "questionCount": 5, "sourceFiles": ["job-description"]},
            {"skill": "SQL", "questionCount": 5, "sourceFiles": ["job-description"]},
        ],
    }
    patch = {
        "replaceCoverage": [
            {
                "skill": "Git",
                "questionCount": 10,
                "focusAreas": ["syntax"],
                "sourceFiles": ["git-internals.pdf"],
            }
        ],
        "replaceSkills": ["Git"],
    }
    merged = _merge_baseline_patch(baseline, patch)
    assert merged["summary"] == "Old summary"
    assert merged["skills"] == ["Git"]
    assert len(merged["coverage"]) == 1
    assert merged["coverage"][0]["skill"] == "Git"


def test_refine_plan_request_baseline_alias():
    payload = {
        "ownerId": "user-1",
        "jobDescription": "JD text",
        "baselinePlan": {"roleTitle": "Dev", "totalQuestions": 5},
        "numberOfQuestions": 5,
        "difficulty": "medium",
    }
    req = RefinePlanRequest.model_validate(payload)
    assert req.baseline_plan["roleTitle"] == "Dev"
