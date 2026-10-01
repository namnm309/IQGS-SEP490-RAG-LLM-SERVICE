"""SCRUM-504: đối chiếu từ khoá công nghệ trong text CV để bù skill LLM bỏ sót.

Vì sao cần: skill CV trước đây chỉ do một lượt LLM trích. Model hay bỏ sót công nghệ
nằm rải rác trong phần project/experience (vd. CV có cả C# và Java nhưng chỉ nhận C#).
Lớp này quét deterministic nên không phụ thuộc may rủi của model.

Nguyên tắc tránh nhận nhầm:
- Match theo ranh giới token: "Java" KHÔNG khớp trong "JavaScript".
- Tên ngắn / dễ trùng tiếng Anh (Go, R, C, Dart…) chỉ nhận khi đứng độc lập
  trong danh sách kỹ năng (đầu dòng, sau dấu phẩy / bullet / gạch dọc).
"""

from __future__ import annotations

import re

# (tên chuẩn, các biến thể cần quét, có dễ trùng từ tiếng Anh không)
_CATALOG: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    (".NET", (".net", "dotnet"), False),
    ("ASP.NET Core", ("asp.net core", "aspnet core", "asp.net-core"), False),
    ("ASP.NET", ("asp.net", "aspnet"), False),
    ("C#", ("c#", "csharp", "c sharp"), False),
    ("EF Core", ("ef core", "efcore", "entity framework core"), False),
    ("Entity Framework", ("entity framework",), False),
    ("LINQ", ("linq",), False),
    ("SignalR", ("signalr",), False),
    ("Blazor", ("blazor",), False),
    ("xUnit", ("xunit",), False),
    ("NUnit", ("nunit",), False),
    ("JWT", ("jwt",), False),
    ("OAuth", ("oauth", "oauth2"), False),
    ("Java", ("java",), False),
    ("Spring Boot", ("spring boot", "springboot"), False),
    ("Spring", ("spring framework",), False),
    ("Hibernate", ("hibernate",), False),
    ("Maven", ("maven",), False),
    ("Kotlin", ("kotlin",), False),
    ("Python", ("python",), False),
    ("Django", ("django",), False),
    ("FastAPI", ("fastapi", "fast api"), False),
    ("Flask", ("flask",), False),
    ("JavaScript", ("javascript", "java script"), False),
    ("TypeScript", ("typescript", "type script"), False),
    ("React", ("react", "reactjs", "react.js"), False),
    ("React Native", ("react native", "react-native"), False),
    ("Next.js", ("next.js", "nextjs"), False),
    ("Vue", ("vue", "vuejs", "vue.js"), False),
    ("Angular", ("angular", "angularjs"), False),
    ("Node.js", ("node.js", "nodejs"), False),
    ("Express", ("express.js", "expressjs"), False),
    ("Express", ("express",), True),
    ("HTML", ("html", "html5"), False),
    ("CSS", ("css", "css3"), False),
    ("Tailwind CSS", ("tailwind css", "tailwindcss", "tailwind"), False),
    ("Go", ("golang",), False),
    ("Go", ("go",), True),
    ("Rust", ("rust",), True),
    ("Swift", ("swift",), True),
    ("Dart", ("dart",), True),
    ("Flutter", ("flutter",), False),
    ("PHP", ("php",), False),
    ("Laravel", ("laravel",), False),
    ("Ruby on Rails", ("ruby on rails", "rails"), False),
    ("Ruby", ("ruby",), True),
    ("Scala", ("scala",), True),
    ("C++", ("c++", "cpp"), False),
    ("C", ("c",), True),
    ("SQL", ("sql",), False),
    ("PostgreSQL", ("postgresql", "postgres"), False),
    ("MySQL", ("mysql",), False),
    ("SQL Server", ("sql server", "sqlserver", "mssql"), False),
    ("Oracle", ("oracle",), False),
    ("MongoDB", ("mongodb", "mongo db"), False),
    ("Redis", ("redis",), False),
    ("Elasticsearch", ("elasticsearch", "elastic search"), False),
    ("Docker", ("docker",), False),
    ("Kubernetes", ("kubernetes", "k8s"), False),
    ("CI/CD", ("ci/cd", "cicd"), False),
    ("Jenkins", ("jenkins",), False),
    ("GitHub Actions", ("github actions",), False),
    ("GitLab CI", ("gitlab ci",), False),
    ("Git", ("git",), False),
    ("AWS", ("aws", "amazon web services"), False),
    ("Azure", ("azure",), False),
    ("GCP", ("gcp", "google cloud"), False),
    ("Terraform", ("terraform",), False),
    ("Linux", ("linux", "ubuntu"), False),
    ("Nginx", ("nginx",), False),
    ("REST API", ("rest api", "restful api", "restful"), False),
    ("GraphQL", ("graphql",), False),
    ("gRPC", ("grpc",), False),
    ("Microservices", ("microservice", "microservices"), False),
    ("Clean Architecture", ("clean architecture",), False),
    ("DDD", ("domain driven design", "domain-driven design"), False),
    ("CQRS", ("cqrs",), False),
    ("Kafka", ("kafka",), False),
    ("RabbitMQ", ("rabbitmq", "rabbit mq"), False),
    ("Firebase", ("firebase",), False),
    ("Unity", ("unity",), False),
    ("Selenium", ("selenium",), False),
    ("Playwright", ("playwright",), False),
    ("Jest", ("jest",), False),
    ("Pytest", ("pytest",), False),
    ("JUnit", ("junit",), False),
    ("Scrum", ("scrum",), False),
    ("Agile", ("agile",), False),
    ("TDD", ("tdd",), False),
    ("OOP", ("oop", "object oriented programming"), False),
    ("Design Patterns", ("design pattern", "design patterns"), False),
    ("System Design", ("system design",), False),
)

# Ký tự được coi là "thuộc tên công nghệ" — dùng cho ranh giới match.
_TOKEN_CHARS = r"A-Za-z0-9#+"
# Ranh giới của một mục trong danh sách kỹ năng.
# Gồm cả ':' và '-' để bắt được dòng dạng "Languages: Go, Python" hay "- Go".
_LIST_SEP = r"[,;:|•·/\-\t\n\r]"


def _build_pattern(variant: str, ambiguous: bool) -> re.Pattern[str]:
    escaped = re.escape(variant)
    if ambiguous:
        # Chỉ nhận khi đứng độc lập như một mục trong list kỹ năng.
        return re.compile(
            rf"(?:^|{_LIST_SEP})\s*{escaped}\s*(?={_LIST_SEP}|$)",
            re.IGNORECASE | re.MULTILINE,
        )
    return re.compile(
        rf"(?<![{_TOKEN_CHARS}]){escaped}(?![{_TOKEN_CHARS}])",
        re.IGNORECASE,
    )


_COMPILED: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (canonical, _build_pattern(variant, ambiguous))
    for canonical, variants, ambiguous in _CATALOG
    for variant in variants
)


def _key(value: str) -> str:
    """Khoá so trùng: bỏ khoảng trắng / gạch, chữ thường."""
    return re.sub(r"[\s_\-]+", "", (value or "").strip().lower())


def find_skills_in_text(text: str) -> list[str]:
    """Các công nghệ trong catalog xuất hiện trong text CV (theo tên chuẩn)."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for canonical, pattern in _COMPILED:
        key = _key(canonical)
        if key in seen:
            continue
        if pattern.search(text):
            seen.add(key)
            found.append(canonical)
    return found


def backfill_skills(llm_skills: list[str], text: str) -> tuple[list[str], list[str]]:
    """Union skill LLM với skill quét được từ text.

    Trả về (danh sách sau khi bù, danh sách skill được bù thêm) để log/trace.
    Giữ nguyên thứ tự và casing của skill LLM — chỉ thêm phần còn thiếu vào cuối.
    """
    merged = list(llm_skills or [])
    have = {_key(s) for s in merged}
    added: list[str] = []
    for skill in find_skills_in_text(text):
        key = _key(skill)
        if key in have:
            continue
        have.add(key)
        merged.append(skill)
        added.append(skill)
    return merged, added
