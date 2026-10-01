"""SCRUM-504: bù skill bị LLM bỏ sót từ text CV, không nhận nhầm tên lồng nhau."""

from helpers.tech_skill_catalog import backfill_skills, find_skills_in_text

CV_TEXT = """
NGUYEN VAN A - Backend Developer

TECHNICAL SKILLS
- Languages: C#, Java, SQL
- Frameworks: ASP.NET Core, Spring Boot, Entity Framework Core
- Database: PostgreSQL, Redis
- DevOps: Docker, Kubernetes, GitHub Actions

PROJECTS
- Order service viết bằng Java 17 + Spring Boot, deploy bằng Docker.
- Internal portal: ASP.NET Core Web API, EF Core, PostgreSQL.
"""


def test_backfill_adds_java_missed_by_llm() -> None:
    """Đúng ca người dùng báo: LLM chỉ trả C#, Java phải được bù lại."""
    merged, added = backfill_skills(["C#", "ASP.NET Core"], CV_TEXT)
    keys = [s.lower() for s in merged]
    assert "java" in keys
    assert "Java" in added
    # Skill LLM giữ nguyên thứ tự + casing
    assert merged[0] == "C#"


def test_backfill_does_not_duplicate_existing() -> None:
    merged, added = backfill_skills(["C#", "java", "Docker"], CV_TEXT)
    assert sum(1 for s in merged if s.lower() == "java") == 1
    assert all(s.lower() != "java" for s in added)


def test_java_keyword_does_not_match_javascript() -> None:
    found = find_skills_in_text("Frontend stack: JavaScript, TypeScript, React")
    assert "JavaScript" in found
    assert "Java" not in found


def test_short_ambiguous_names_need_list_context() -> None:
    """"go" trong câu văn tiếng Anh không được tính là ngôn ngữ Go."""
    assert "Go" not in find_skills_in_text("I always go the extra mile for my team.")
    assert "Go" in find_skills_in_text("Languages: Go, Python, Java")
    assert "Go" in find_skills_in_text("Backend written in Golang and deployed on AWS")


def test_c_and_cpp_are_separate() -> None:
    found = find_skills_in_text("Languages: C, C++, C#")
    assert "C" in found
    assert "C++" in found
    assert "C#" in found
