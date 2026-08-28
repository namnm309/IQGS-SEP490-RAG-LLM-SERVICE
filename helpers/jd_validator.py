import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class JdValidationStats:
    char_count: int = 0
    word_count: int = 0
    meaningful_lines: int = 0
    signal_groups_matched: int = 0


@dataclass
class JdValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: JdValidationStats = field(default_factory=JdValidationStats)


SIGNAL_GROUPS = [
    ("vai trò", re.compile(
        r"job\s*description|position|role|vị\s*trí|nhân\s*viên|tuyển\s*dụng",
        re.IGNORECASE,
    )),
    ("yêu cầu", re.compile(
        r"must[\s-]*have|requirement|yêu\s*cầu|qualification|kỹ\s*năng|kinh\s*nghiệm",
        re.IGNORECASE,
    )),
    ("trách nhiệm", re.compile(
        r"responsibilit|nhiệm\s*vụ|mô\s*tả\s*công\s*việc|trách\s*nhiệm",
        re.IGNORECASE,
    )),
    ("level", re.compile(
        r"\bSWE\b|senior|junior|mid[\s-]*level|\byears?\b|năm\s*kinh\s*nghiệm|\blevel\b",
        re.IGNORECASE,
    )),
]

# SCRUM-416: domain IT — đồng bộ với BE JobDescriptionValidator (keyword bilingual).
IT_ROLE_KEYWORDS = [
    "software engineer", "software developer", "backend", "front-end", "frontend", "full stack",
    "fullstack", "full-stack", "devops", "sre", "site reliability", "data engineer", "data scientist",
    "machine learning", "ml engineer", "ai engineer", "qa engineer", "test engineer", "quality assurance",
    "mobile developer", "ios developer", "android developer", "security engineer", "cybersecurity",
    "cloud engineer", "platform engineer", "database administrator", "dba", "system admin", "sysadmin",
    "it support", "technical lead", "tech lead", "solution architect", "software architect", "embedded",
    "lập trình viên", "kỹ sư phần mềm", "nhà phát triển", "phát triển phần mềm", "lập trình",
    "kiểm thử phần mềm", "kỹ sư qa", "kỹ sư dữ liệu",
]

IT_TECH_KEYWORDS = [
    ".net", "asp.net", "c#", "csharp", "java", "spring boot", "python", "django", "fastapi", "flask",
    "javascript", "typescript", "react", "next.js", "nextjs", "angular", "vue", "node.js", "nodejs",
    "golang", "go lang", "rust", "kotlin", "swift", "php", "laravel", "ruby on rails",
    "sql", "postgresql", "postgres", "mysql", "mongodb", "redis", "elasticsearch",
    "docker", "kubernetes", "k8s", "ci/cd", "jenkins", "github actions", "gitlab ci",
    "aws", "azure", "gcp", "google cloud", "terraform", "ansible",
    "rest api", "graphql", "microservices", "kafka", "rabbitmq", "grpc",
    "git", "linux", "api gateway", "unit test", "xunit", "junit", "pytest",
    "html", "css", "tailwind", "webpack",
]

NON_IT_KEYWORDS = [
    "marketing", "digital marketing", "content marketing", "seo specialist", "social media",
    "sales executive", "sales manager", "account executive", "business development",
    "kế toán", "accountant", "accounting", "bookkeeper", "kiểm toán", "auditor",
    "luật sư", "lawyer", "legal counsel", "pháp chế",
    "giáo viên", "teacher", "giảng viên", "nhân viên y tế", "bác sĩ", "nurse", "điều dưỡng",
    "nhà hàng", "restaurant", "pha chế", "bartender", "khách sạn", "hotel receptionist",
    "bất động sản", "real estate", "môi giới",
    "nhân viên bán hàng", "cashier", "thu ngân", "warehouse picker",
    "lễ tân", "receptionist", "fashion designer", "thiết kế thời trang", "makeup artist",
]


def _score_keywords(lower: str, keywords: list[str]) -> int:
    return sum(1 for kw in keywords if kw in lower)


def validate_it_domain(text: str) -> str | None:
    """Trả về message lỗi nếu JD không thuộc IT; None nếu OK."""
    lower = (text or "").lower()
    it_score = _score_keywords(lower, IT_ROLE_KEYWORDS) + _score_keywords(lower, IT_TECH_KEYWORDS)
    non_it_score = _score_keywords(lower, NON_IT_KEYWORDS)

    if it_score == 0:
        return (
            "Hệ thống chỉ nhận Job Description thuộc lĩnh vực IT/phần mềm. "
            "JD này không có tín hiệu kỹ thuật đủ rõ (vai trò IT hoặc công nghệ như .NET, React, Java, SQL…)."
        )

    if non_it_score > it_score:
        return (
            "Hệ thống chỉ nhận Job Description thuộc lĩnh vực IT/phần mềm. "
            "JD này nghiêng về ngành ngoài IT (ví dụ marketing, sales, kế toán, luật…). "
            "Vui lòng dùng JD kỹ thuật/phần mềm."
        )

    return None


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _normalize_lines(text: str) -> List[str]:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return lines


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    return printable / len(text)


def _duplicate_line_ratio(lines: List[str]) -> float:
    if len(lines) < 2:
        return 0.0
    unique = len(set(lines))
    return 1.0 - (unique / len(lines))


def _is_trivial_bullet_line(line: str) -> bool:
    cleaned = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
    return 0 < len(cleaned.split()) <= 2 and len(cleaned) < 25


def validate_jd_text(text: str, file_name: str, config: dict) -> JdValidationResult:
    min_chars = config.get("jd_min_chars", 400)
    max_chars = config.get("jd_max_chars", 30_000)
    min_words = config.get("jd_min_words", 100)
    max_words = config.get("jd_max_words", 5_000)

    normalized = (text or "").strip()
    stats = JdValidationStats(
        char_count=len(normalized),
        word_count=_count_words(normalized),
    )
    errors: List[str] = []
    warnings: List[str] = []
    label = file_name or "JD"

    if not normalized:
        errors.append(f"File '{label}' không có nội dung text sau khi trích xuất.")
        return JdValidationResult(valid=False, errors=errors, stats=stats)

    if _printable_ratio(normalized) < 0.85:
        errors.append(
            f"File '{label}' có quá nhiều ký tự không đọc được. "
            "Vui lòng kiểm tra lại định dạng hoặc xuất lại file."
        )

    if stats.char_count < min_chars:
        errors.append(
            f"JD quá ngắn ({stats.char_count} ký tự, tối thiểu {min_chars}). "
            "Vui lòng bổ sung mô tả vai trò, yêu cầu và trọng tâm phỏng vấn."
        )
    elif stats.char_count > max_chars:
        errors.append(
            f"JD quá dài ({stats.char_count:,} ký tự, tối đa {max_chars:,}). "
            "Vui lòng tách file hoặc rút gọn phần không liên quan."
        )

    if stats.word_count < min_words:
        errors.append(
            f"JD quá ngắn ({stats.word_count} từ, tối thiểu {min_words}). "
            "Vui lòng bổ sung chi tiết về vai trò và yêu cầu công việc."
        )
    elif stats.word_count > max_words:
        errors.append(
            f"JD quá dài ({stats.word_count:,} từ, tối đa {max_words:,}). "
            "Vui lòng rút gọn hoặc tách thành nhiều file."
        )

    lines = _normalize_lines(normalized)
    stats.meaningful_lines = len(lines)

    dup_ratio = _duplicate_line_ratio(lines)
    if dup_ratio > 0.6:
        errors.append(
            "JD có quá nhiều dòng trùng lặp (>60%). "
            "Vui lòng loại bỏ nội dung copy-paste không cần thiết."
        )

    trivial_bullets = sum(1 for line in lines if _is_trivial_bullet_line(line))
    if lines and trivial_bullets / len(lines) > 0.7:
        errors.append(
            "JD chủ yếu là bullet quá ngắn, thiếu mô tả cụ thể. "
            "Vui lòng bổ sung chi tiết về vai trò, kỹ năng và trọng tâm phỏng vấn."
        )

    matched_groups = []
    for group_name, pattern in SIGNAL_GROUPS:
        if pattern.search(normalized):
            matched_groups.append(group_name)
    stats.signal_groups_matched = len(matched_groups)

    if stats.signal_groups_matched < 2:
        errors.append(
            "Thiếu thông tin cốt lõi trong JD: cần có ít nhất 2 trong 4 nhóm "
            "(vai trò, yêu cầu, trách nhiệm, level/kinh nghiệm). "
            f"Hiện chỉ nhận diện được: {', '.join(matched_groups) or 'không có'}."
        )

    if stats.signal_groups_matched == 2:
        warnings.append(
            "JD chỉ đáp ứng tối thiểu 2/4 nhóm thông tin. "
            "Nên bổ sung thêm chi tiết để plan chính xác hơn."
        )

    # SCRUM-416: chỉ nhận JD IT/phần mềm (đồng bộ BE).
    it_error = validate_it_domain(normalized)
    if it_error:
        errors.append(it_error)

    return JdValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )


def load_combined_hr_jd_text(data_folder: str, owner_id: str, extract_text_fn) -> str:
    import os
    from pathlib import Path

    folder = os.path.join(os.getcwd(), data_folder, "hr", owner_id)
    if not os.path.isdir(folder):
        return ""

    supported = {".pdf", ".docx", ".txt", ".md", ".json", ".xlsx", ".xls"}
    parts: List[str] = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() not in supported:
            continue
        try:
            text = extract_text_fn(str(path))
            if text and text.strip():
                parts.append(f"--- {path.name} ---\n{text.strip()}")
        except Exception:
            continue
    return "\n\n".join(parts)
