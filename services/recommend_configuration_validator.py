"""Validate + normalize AI interview configuration recommendation."""
from __future__ import annotations

from typing import Any

CANONICAL_CATEGORIES = frozenset({"technical", "behavioral", "situational"})
CANONICAL_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
CANONICAL_STYLES = frozenset(
    {
        "system_design",
        "problem_solving",
        "debugging",
        "performance_analysis",
        "coding",
        "code_review",
        "theory",
    }
)
CANONICAL_CODING_TASK_TYPES = frozenset(
    {
        "BUG_DETECTION",
        "CODE_COMPLETION",
        "REFACTORING",
        "TEST_CASE_DESIGN",
        "PERFORMANCE_ANALYSIS",
    }
)

_STYLE_ALIASES = {
    "system design": "system_design",
    "system-design": "system_design",
    "systemdesign": "system_design",
    "problem solving": "problem_solving",
    "problem-solving": "problem_solving",
    "problemsolving": "problem_solving",
    "performance": "performance_analysis",
    "performance analysis": "performance_analysis",
    "performance-analysis": "performance_analysis",
    "code review": "code_review",
    "code-review": "code_review",
    "codereview": "code_review",
}

_CATEGORY_ALIASES = {
    "technical": "technical",
    "behavioral": "behavioral",
    "situational": "situational",
    "system_design": "technical",
    "system-design": "technical",
    "problem_solving": "technical",
    "problem-solving": "technical",
}


def _norm_key(value: str) -> str:
    return (value or "").strip().lower().replace(" ", "_").replace("-", "_")


def normalize_category(value: str | None) -> str:
    if not value:
        return "technical"
    key = _norm_key(value)
    mapped = _CATEGORY_ALIASES.get(key.replace("_", "-")) or _CATEGORY_ALIASES.get(key)
    if mapped:
        return mapped
    if key in CANONICAL_CATEGORIES:
        return key
    return "technical"


def normalize_difficulty(value: str | None) -> str:
    if not value:
        return "medium"
    key = _norm_key(value)
    return key if key in CANONICAL_DIFFICULTIES else "medium"


def normalize_style(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip().lower()
    if raw in _STYLE_ALIASES:
        return _STYLE_ALIASES[raw]
    key = _norm_key(value)
    if key in CANONICAL_STYLES:
        return key
    return None


def normalize_coding_task_type(value: str | None) -> str | None:
    if not value:
        return None
    key = value.strip().upper().replace(" ", "_")
    if key == "SYSTEM_DESIGN":
        return None
    return key if key in CANONICAL_CODING_TASK_TYPES else None


def _largest_remainder_percentages(counts: dict[str, int], total: int) -> dict[str, int]:
  if total <= 0:
      return {k: 0 for k in counts}
  raw = {k: (v / total) * 100 for k, v in counts.items()}
  floors = {k: int(raw[k]) for k in raw}
  remainder = 100 - sum(floors.values())
  order = sorted(counts.keys(), key=lambda k: (raw[k] - floors[k]), reverse=True)
  for i in range(remainder):
      floors[order[i % len(order)]] += 1
  return floors


def validate_and_normalize_recommendation(
    data: dict[str, Any],
    *,
    default_total: int = 10,
) -> tuple[dict[str, Any], list[str]]:
    """Return normalized recommendation dict and validation errors (empty if ok)."""
    errors: list[str] = []
    cfg = data.get("recommendedConfiguration") or data.get("recommended_configuration") or data

    total = int(cfg.get("numberOfQuestions") or cfg.get("number_of_questions") or default_total)
    total = max(1, min(50, total))
    difficulty = normalize_difficulty(cfg.get("difficulty"))

    dist_raw = cfg.get("questionDistribution") or cfg.get("question_distribution") or []
    category_counts: dict[str, int] = {"technical": 0, "behavioral": 0, "situational": 0}
    styles_from_legacy: list[str] = []

    if isinstance(dist_raw, list) and dist_raw:
        for item in dist_raw:
            if not isinstance(item, dict):
                continue
            cat = normalize_category(str(item.get("category") or item.get("type") or "technical"))
            count = int(item.get("questionCount") or item.get("question_count") or 0)
            if count <= 0:
                pct = int(item.get("percentage") or 0)
                count = max(0, round(total * pct / 100)) if pct else 0
            category_counts[cat] = category_counts.get(cat, 0) + count
            legacy_style = normalize_style(str(item.get("category") or item.get("type") or ""))
            if legacy_style and legacy_style not in ("technical", "behavioral", "situational"):
                styles_from_legacy.append(legacy_style)
    else:
        category_counts = {"technical": max(1, int(total * 0.6)), "behavioral": max(0, int(total * 0.2)), "situational": 0}
        category_counts["situational"] = total - category_counts["technical"] - category_counts["behavioral"]

    active = {k: v for k, v in category_counts.items() if v > 0}
    if not active:
        active = {"technical": total}

    sum_counts = sum(active.values())
    if sum_counts != total:
        scale_keys = list(active.keys())
        adjusted: dict[str, int] = {k: 0 for k in category_counts}
        remaining = total
        for i, k in enumerate(scale_keys):
            if i == len(scale_keys) - 1:
                adjusted[k] = remaining
            else:
                c = max(1, round(active[k] * total / sum_counts)) if sum_counts else 1
                c = min(c, remaining - (len(scale_keys) - i - 1))
                adjusted[k] = c
                remaining -= c
        active = {k: v for k, v in adjusted.items() if v > 0}

    pcts = _largest_remainder_percentages(active, total)
    question_distribution = [
        {
            "category": cat,
            "percentage": pcts[cat],
            "questionCount": active[cat],
        }
        for cat in ("technical", "behavioral", "situational")
        if cat in active
    ]

    styles_raw = cfg.get("questionStyles") or cfg.get("question_styles") or []
    styles: list[str] = []
    if isinstance(styles_raw, list):
        for s in styles_raw:
            norm = normalize_style(str(s))
            if norm and norm not in styles:
                styles.append(norm)
    for s in styles_from_legacy:
        if s not in styles:
            styles.append(s)

    coding_raw = cfg.get("codingTaskTypes") or cfg.get("coding_task_types") or []
    coding: list[str] = []
    if isinstance(coding_raw, list):
        for c in coding_raw:
            norm = normalize_coding_task_type(str(c))
            if norm and norm not in coding:
                coding.append(norm)

    coding_recommended = bool(cfg.get("codingTasksRecommended") or cfg.get("coding_tasks_recommended"))
    if coding:
        coding_recommended = True

    focus_raw = cfg.get("focusAreas") or cfg.get("focus_areas") or []
    focus_areas: list[dict[str, Any]] = []
    if isinstance(focus_raw, list):
        for idx, item in enumerate(focus_raw):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            weight = float(item.get("weight") or 0)
            focus_areas.append(
                {
                    "name": name[:150],
                    "weight": weight,
                    "description": (str(item.get("description") or "").strip() or None),
                    "sourceReason": (str(item.get("sourceReason") or item.get("source_reason") or "").strip() or None),
                    "orderIndex": int(item.get("orderIndex") or item.get("order_index") or idx),
                }
            )

    if focus_areas:
        wsum = sum(f["weight"] for f in focus_areas)
        if wsum <= 0:
            even = round(100.0 / len(focus_areas), 2)
            for f in focus_areas:
                f["weight"] = even
        elif abs(wsum - 100) > 0.5:
            factor = 100.0 / wsum
            for f in focus_areas:
                f["weight"] = round(float(f["weight"]) * factor, 2)

    if not focus_areas:
        errors.append("focusAreas must not be empty")

    normalized = {
        "numberOfQuestions": total,
        "difficulty": difficulty,
        "questionDistribution": question_distribution,
        "focusAreas": focus_areas,
        "questionStyles": styles,
        "codingTaskTypes": coding,
        "codingTasksRecommended": coding_recommended,
    }
    return normalized, errors
