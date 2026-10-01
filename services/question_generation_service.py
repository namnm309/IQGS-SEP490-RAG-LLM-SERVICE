"""Sinh câu hỏi phỏng vấn từ JD hoặc approved plan — JSON only output."""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from typing import Any

from openai import OpenAI

from config.settings import Settings
from models.internal_schemas import (
    VALID_QUESTION_TYPES,
    FlaggedQuestionItem,
    GeneratedQuestionItem,
    GenerateQuestionsFromPlanRequest,
    GenerateQuestionsFromPlanResponse,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    QuestionGenerationPlan,
    RubricCriterionItem,
    normalize_question_type,
)
from services.json_output_parser import (
    build_json_fix_prompt,
    extract_json_object,
    is_retryable_json_error,
)
from services.rag_context_helpers import (
    build_chunk_lookup,
    enrich_citations,
    ensure_jd_primary_citations,
    format_retrieved_context,
    is_jd_source_file,
    parse_citations,
    JD_SOURCE_FILE,
)
from services.rag_retrieval_service import RagRetrievalService
from services.question_provenance_validator import apply_provenance_to_questions
from services.rag_context_helpers import is_jd_source_file
from vectorstores.base import RetrievedChunk
from helpers.language_prompt import (
    free_text_language_label,
    language_instruction_block,
    normalize_output_language,
)

logger = logging.getLogger(__name__)

# SCRUM-495 harden: retry sinh lại khi lệch config HR
_LABEL_MISMATCH_MAX_RETRIES = 3
_DISTRIBUTION_FIX_MAX_ROUNDS = 2

_DEFAULT_CODE_TEMPLATES = [
    "CODE_COMPLETION",
    "BUG_DETECTION",
    "REFACTORING",
    "TEST_CASE_DESIGN",
    "PERFORMANCE_ANALYSIS",
    "SYSTEM_DESIGN",
]

# Template bắt buộc có code_snippet thật (không placeholder).
_TEMPLATES_REQUIRING_SNIPPET = frozenset(
    {
        "CODE_COMPLETION",
        "BUG_DETECTION",
        "REFACTORING",
        "TEST_CASE_DESIGN",
        "PERFORMANCE_ANALYSIS",
    }
)

# SCRUM-396: rule chi tiết theo template — inject vào user prompt (chỉ enabled).
_TEMPLATE_RULES_BY_ID: dict[str, str] = {
    "CODE_COMPLETION": """TEMPLATE CODE_COMPLETION — Hoàn thiện mã nguồn
Mục tiêu: đánh giá đọc hiểu và hoàn thiện code.
Yêu cầu:
- Sinh đoạn code chưa hoàn chỉnh; chỉ thiếu DUY NHẤT một phần quan trọng.
- Không lỗi cú pháp; không comment/tên biến tiết lộ đáp án.
- Candidate tự hoàn thành phần còn thiếu.
- Độ dài khoảng 15–30 dòng; chủ đề sát thực tế / JD.
- code_snippet BẮT BUỘC = skeleton code thật đúng ngôn ngữ JD (C#/SQL/JS/…), KHÔNG stub.
image_hint: không bắt buộc; có thể gợi ý Flowchart hoặc Sequence Diagram. KHÔNG gen ảnh.""",
    "BUG_DETECTION": """TEMPLATE BUG_DETECTION — Tìm lỗi logic
Mục tiêu: đọc code, phân tích lỗi, debug.
Yêu cầu:
- Code biên dịch được; chỉ DUY NHẤT một lỗi logic; không lỗi cú pháp; không tiết lộ vị trí lỗi.
- Candidate: xác định lỗi + nguyên nhân + cách sửa.
- Độ dài khoảng 15–30 dòng.
- code_snippet BẮT BUỘC = đoạn code có bug logic thật.
image_hint: có thể gợi ý Flowchart hoặc Activity Diagram. KHÔNG gen ảnh.""",
    "REFACTORING": """TEMPLATE REFACTORING — Cải thiện chất lượng code
Mục tiêu: readability / maintainability / duplicate / naming.
Yêu cầu:
- Code đang chạy đúng; có smell (đọc/bảo trì/trùng lặp/đặt tên); KHÔNG tạo bug.
- Candidate refactor nhưng không đổi chức năng; không tiết lộ hướng refactor.
- code_snippet BẮT BUỘC = code “xấu nhưng đúng” cần refactor.
image_hint: có thể gợi ý Class Diagram hoặc Dependency Diagram. KHÔNG gen ảnh.""",
    "TEST_CASE_DESIGN": """TEMPLATE TEST_CASE_DESIGN — Thiết kế kiểm thử
Mục tiêu: thiết kế test case.
Yêu cầu:
- Sinh hàm hoặc API + mô tả yêu cầu rõ; KHÔNG sinh sẵn test case / đáp án.
- Candidate thiết kế: Normal + Boundary + Invalid.
- code_snippet BẮT BUỘC = chữ ký hàm/API hoặc skeleton cần test (không kèm test).
image_hint: có thể gợi ý Use Case Diagram hoặc Activity Diagram. KHÔNG gen ảnh.""",
    "PERFORMANCE_ANALYSIS": """TEMPLATE PERFORMANCE_ANALYSIS — Phân tích hiệu năng
Mục tiêu: Time/Space complexity, bottleneck, tối ưu.
Yêu cầu:
- Code chạy đúng nhưng chưa tối ưu; không tiết lộ đáp án trong đề.
- Candidate: Time Complexity + Space Complexity + Bottleneck + phương án tối ưu.
- code_snippet BẮT BUỘC = đoạn code có điểm chậm cần phân tích.
image_hint: có thể gợi ý Performance/Benchmark Chart hoặc Flowchart. KHÔNG gen ảnh.""",
    "SYSTEM_DESIGN": """TEMPLATE SYSTEM_DESIGN — Thiết kế hệ thống
Mục tiêu: tư duy kiến trúc hệ thống thực tế (URL shortener, chat, delivery, e-commerce, interview platform, streaming…).
Yêu cầu:
- Bài toán thiết kế rõ; candidate trình bày: kiến trúc, thành phần, DB, cache, MQ (nếu cần), scalability, trade-off.
- Không tiết lộ đáp án trong câu hỏi.
- code_snippet OPTIONAL (có thể để trống hoặc gợi ý text diagram ngắn).
image_hint: KHUYẾN NGHỊ Architecture / C4 / Deployment / sơ đồ tổng quan. KHÔNG gen ảnh.""",
}

_TEMPLATE_COMMON_RULES = """QUY TẮC CHUNG (batch generate):
- Câu lý thuyết (Git PR, workflow, khái niệm) → answer_method=Text, KHÔNG code_snippet, KHÔNG CODE_COMPLETION.
- Chỉ câu bài code (hoàn thiện/bug/refactor) mới gắn code_template_type + code_snippet đúng skill/focus.
- code_snippet = ĐỀ BÀI (đoạn code ứng viên phải đọc). BẮT BUỘC với template CODE_COMPLETION/BUG_DETECTION/REFACTORING/TEST_CASE_DESIGN/PERFORMANCE_ANALYSIS — kể cả khi answer_method=Text (vd. tìm bug rồi giải thích bằng chữ).
- KHÔNG để code lỗi/skeleton chỉ trong sample_answer mà thiếu code_snippet đề.
- code_snippet phải cùng chủ đề câu hỏi (Git → lệnh git/bash; không dán C# OrderService vào câu Git).
- Không Multiple Choice.
- Nội dung phù hợp Programming Language (từ JD/skills), Difficulty, Skills, JD, RAG context.
- Không tiết lộ đáp án trong question / code_snippet.
- Template cần code → code_snippet thật đúng chủ đề (không placeholder, không snippet generic lệch skill).
- image_hint chỉ gợi ý loại hình HR nên tải — tuyệt đối không sinh hình ảnh.
- Trong JSON string: escape đúng \\\\n thành newline thật sau parse — không double-escape thành chữ \\\\n trên UI.
- sample_answer (khi có code đáp án) BẮT BUỘC tách text/code trong GIÁ TRỊ string:
  (1) 1–3 câu giải thích văn xuôi TRƯỚC,
  (2) rồi một markdown fence đúng ngôn ngữ, ví dụ:
      ```csharp
      // code đáp án
      ```
  Không viết "code:" rồi dán code liền một khối. Không nhét giải thích tiếng Việt vào trong fence.
  Không copy code_snippet (đề bài) vào sample_answer.
- Trả về đúng Output Schema hệ thống yêu cầu."""

_INTERVIEW_SYSTEM_PROMPT_BASE = """Bạn là chuyên gia thiết kế câu hỏi phỏng vấn kỹ thuật.

## Mục tiêu — Waterfall JD → Admin → LLM
1. **JD (HR)**: *Vì sao* có câu này — citation JD đầu tiên, excerpt nguyên văn skill/yêu cầu.
2. **Admin ([HỆ THỐNG])**: *Kiến thức kỹ thuật chuẩn* — câu technical/problem-solving/system-design PHẢI cite chunk [HỆ THỐNG] nếu context có.
3. **LLM**: *Bổ sung cuối* — sample_answer/rubric không map chunk → origin=LLM + reason.

## Quy tắc
1. Bám JD cho lý do hỏi. Không bịa yêu cầu không có trong JD.
2. Mỗi câu hỏi phải bám difficulty và skills được yêu cầu.
3. Cân bằng loại câu hỏi theo question_types được yêu cầu.
4. {language_rule}
5. Chỉ trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON.
6. Mỗi câu BẮT BUỘC citation JD đầu tiên: knowledge_base=\"hr\", source_file=\"{jd_source}\", origin=\"HR\", usedFor=[\"why-asked\"]. SCRUM-425: chunk_index = đoạn JD khớp skill/yêu cầu của câu (không luôn 0); excerpt nguyên văn đoạn đó.
7. SCRUM-394: excerpt JD BẮT BUỘC trích NGUYÊN VĂN từ jobDescription — không paraphrase, không rỗng; mỗi câu trích đoạn khác nhau nếu skill khác nhau.
8. Citation [HỆ THỐNG] sau JD khi dùng chunk technical: origin=\"SYSTEM\", usedFor=[\"technical-body\"]; excerpt nguyên văn từ chunk.
9. Phần sample_answer/rubric không có chunk → thêm mục origin=\"LLM\" + reason (trong citations hoặc ghi rõ trong JSON).
10. SCRUM-400: mỗi câu BẮT BUỘC có answer_method = \"Text\" | \"Code\".

## Schema JSON bắt buộc
{{
  "questions": [
    {{
      "question": "string",
      "question_type": "technical|behavioral|situational|system-design|problem-solving",
      "difficulty": "easy|medium|hard",
      "rationale": "string",
      "sample_answer": "string",
      "image_hint": "string",
      "answer_method": "Text|Code",
      "citations": [
        {{
          "knowledge_base": "hr",
          "source_file": "{jd_source}",
          "chunk_index": 1,
          "excerpt": "doan trich ngan tu JD khop skill",
          "origin": "HR",
          "usedFor": ["why-asked"]
        }},
        {{
          "knowledge_base": "system",
          "source_file": "ten-file.pdf",
          "chunk_index": 0,
          "excerpt": "doan trich tu chunk he thong",
          "origin": "SYSTEM",
          "usedFor": ["technical-body"]
        }}
      ]
    }}
  ]
}}""".replace("{jd_source}", JD_SOURCE_FILE)

_QUESTIONS_FROM_PLAN_SYSTEM_PROMPT_BASE = """Bạn là chuyên gia sinh câu hỏi phỏng vấn (interview question generator).

## Mục tiêu — Waterfall JD → Admin → LLM
Sinh câu hỏi từ APPROVED PLAN (KHÔNG thay đổi plan):
1. **JD**: citation đầu — excerpt = lý do HR hỏi (skill/requirement), origin=HR, usedFor=[why-asked].
2. **Admin [HỆ THỐNG]**: câu technical/problem-solving/code → PHẢI cite chunk [HỆ THỐNG] nếu context có; origin=SYSTEM, usedFor=[technical-body].
3. **LLM**: sample_answer/rubric không map chunk → origin=LLM + reason. Thiếu Admin doc vẫn sinh câu (soft_llm).

## Quy tắc
1. Tuân thủ approvedPlan: totalQuestions, questionTypeDistribution, coverage, recommendedQuestionOutline.
2. Chỉ trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON.
3. {language_rule}
4. Mỗi câu BẮT BUỘC citation JD đầu tiên (origin HR, usedFor why-asked). SCRUM-425: chunk_index = đoạn JD khớp skill/focus của câu (không luôn 0).
5. SCRUM-394: excerpt JD trích NGUYÊN VĂN từ jobDescription — không paraphrase; mỗi câu trích đoạn khác nếu skill khác.
6. Citation [HỆ THỐNG] sau JD khi dùng chunk; excerpt nguyên văn từ chunk.
7. SCRUM-396/400/418: code_template, answer_method, evaluation_criteria như trước.
8. Thiếu chunk [HỆ THỐNG]: vẫn sinh câu; gắn LLM + reason cho phần bổ sung.
9. Nếu hrNote có STRICT_FOCUS=1: CHỈ hỏi các FOCUS_AREAS. skill/focus_area mỗi câu phải thuộc list. JD = ngữ cảnh vị trí, KHÔNG sinh câu về skill khác trong JD.

## Schema JSON bắt buộc
{{
  "questions": [
    {{
      "order": 1,
      "question": "string",
      "question_type": "technical|behavioral|situational|system-design|problem-solving",
      "difficulty": "easy|medium|hard",
      "skill": "string",
      "focus_area": "string",
      "rationale": "string",
      "sample_answer": "string",
      "evaluation_criteria": [{{ "id": "accuracy", "label": "string", "weight": 40, "anchors": {{ "25": "...", "50": "...", "75": "...", "100": "..." }} }}],
      "code_template_type": "CODE_COMPLETION|...",
      "code_snippet": "string",
      "image_hint": "string",
      "answer_method": "Text|Code",
      "citations": [
        {{
          "knowledge_base": "hr",
          "source_file": "{jd_source}",
          "chunk_index": 1,
          "excerpt": "doan trich tu JD khop skill",
          "origin": "HR",
          "usedFor": ["why-asked"]
        }},
        {{
          "knowledge_base": "system",
          "source_file": "git-guide.pdf",
          "chunk_index": 0,
          "excerpt": "doan trich tu chunk he thong",
          "origin": "SYSTEM",
          "usedFor": ["technical-body"]
        }}
      ]
    }}
  ]
}}""".replace("{jd_source}", JD_SOURCE_FILE)


def _interview_system_prompt(language: str | None) -> str:
    return _INTERVIEW_SYSTEM_PROMPT_BASE.format(
        language_rule=language_instruction_block(language)
    )


def _questions_from_plan_system_prompt(language: str | None) -> str:
    return _QUESTIONS_FROM_PLAN_SYSTEM_PROMPT_BASE.format(
        language_rule=language_instruction_block(language)
    )


# Tương thích import/test cũ
INTERVIEW_SYSTEM_PROMPT = _interview_system_prompt("Vietnamese")
QUESTIONS_FROM_PLAN_SYSTEM_PROMPT = _questions_from_plan_system_prompt("Vietnamese")


def _parse_content_preferences(hr_note: str | None) -> tuple[str, list[str]]:
    note = (hr_note or "").strip()
    if not note:
        return "Mixed", _DEFAULT_CODE_TEMPLATES
    mode_match = re.search(r"CONTENT_MODE=([A-Za-z]+)", note)
    mode = (mode_match.group(1) if mode_match else "Mixed").strip()
    if mode not in {"TheoryOnly", "CodeOnly", "Mixed"}:
        mode = "Mixed"
    tpl_match = re.search(r"CODE_TEMPLATES=([^;\n]+)", note)
    if not tpl_match:
        return mode, _DEFAULT_CODE_TEMPLATES
    templates = [x.strip() for x in tpl_match.group(1).split(",") if x.strip()]
    return mode, templates or _DEFAULT_CODE_TEMPLATES


def _format_enabled_template_rules(templates: list[str]) -> str:
    """Chỉ inject rule của template đang enabled + quy tắc chung (batch)."""
    blocks: list[str] = [
        "[SCRUM-396 TEMPLATE RULES — áp dụng theo code_template_type của từng câu]",
        _TEMPLATE_COMMON_RULES,
        "",
    ]
    seen: set[str] = set()
    for raw in templates:
        tid = (raw or "").strip().upper()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        rule = _TEMPLATE_RULES_BY_ID.get(tid)
        if rule:
            blocks.append(rule)
            blocks.append("")
    if len(seen) == 0:
        for tid in _DEFAULT_CODE_TEMPLATES:
            blocks.append(_TEMPLATE_RULES_BY_ID[tid])
            blocks.append("")
    return "\n".join(blocks).rstrip()


_PLACEHOLDER_SNIPPET_RE = re.compile(
    r"(vi[eế]t\s+(query|code|tại\s*đây)|write\s+(query|code)\s+here|"
    r"complete\s+here|your\s+code\s+here|todo\s*only|"
    r"#\s*TODO\s*:?\s*$|//\s*TODO\s*:?\s*$)",
    re.IGNORECASE,
)


def _unescape_snippet_escapes(snippet: str) -> str:
    """Nếu LLM để chữ \\n/\\t thay vì newline thật → chuyển về ký tự thật."""
    if not snippet:
        return snippet
    if "\n" not in snippet and ("\\n" in snippet or "\\t" in snippet):
        return snippet.replace("\\n", "\n").replace("\\t", "\t")
    # Double-escaped còn sót: \\n hiển thị khi đã có newline lẫn literal
    if "\\n" in snippet and snippet.count("\\n") >= snippet.count("\n"):
        return snippet.replace("\\n", "\n").replace("\\t", "\t")
    return snippet


def _is_placeholder_snippet(snippet: str | None) -> bool:
    """Stub quá ngắn / chỉ comment / chứa cụm placeholder."""
    if not snippet or not snippet.strip():
        return True
    text = snippet.strip()
    if _PLACEHOLDER_SNIPPET_RE.search(text):
        return True
    # Chỉ 1–2 dòng comment / trống nghĩa
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 1 and (
        lines[0].startswith("--")
        or lines[0].startswith("//")
        or lines[0].startswith("#")
        or len(lines[0]) < 40
    ):
        return True
    # Quá ngắn và gần như không có nội dung code
    if len(text) < 40 and len(lines) <= 2:
        return True
    return False


def _harden_code_snippet(snippet: str | None, template_id: str | None) -> str | None:
    """Unescape + reject placeholder; trả None nếu không dùng được."""
    if not snippet:
        return None
    cleaned = _unescape_snippet_escapes(snippet).strip()
    if _is_placeholder_snippet(cleaned):
        logger.info(
            "Bỏ code_snippet placeholder (template=%s, preview=%r)",
            template_id,
            cleaned[:80],
        )
        return None
    return cleaned


def _default_snippet_for_template(template_id: str) -> str:
    samples = {
        "CODE_COMPLETION": (
            "public class OrderService\n"
            "{\n"
            "    private readonly IOrderRepository _orders;\n"
            "    private readonly ICustomerRepository _customers;\n\n"
            "    public OrderService(IOrderRepository orders, ICustomerRepository customers)\n"
            "    {\n"
            "        _orders = orders;\n"
            "        _customers = customers;\n"
            "    }\n\n"
            "    public async Task<decimal> GetCustomerOrderTotalAsync(int customerId)\n"
            "    {\n"
            "        var customer = await _customers.GetByIdAsync(customerId);\n"
            "        if (customer == null) throw new InvalidOperationException(\"Not found\");\n\n"
            "        // Candidate: hoàn thiện phần tính tổng đơn hàng của customer\n"
            "        throw new NotImplementedException();\n"
            "    }\n"
            "}"
        ),
        "BUG_DETECTION": (
            "public int SumPositive(int[] nums)\n"
            "{\n"
            "    int total = 0;\n"
            "    for (int i = 0; i <= nums.Length; i++)\n"
            "    {\n"
            "        if (nums[i] > 0) total += nums[i];\n"
            "    }\n"
            "    return total;\n"
            "}"
        ),
        "REFACTORING": (
            "public string GetDisplayName(User user)\n"
            "{\n"
            "    if (user != null)\n"
            "    {\n"
            "        if (user.Profile != null)\n"
            "        {\n"
            "            if (user.Profile.Name != null)\n"
            "            {\n"
            "                return user.Profile.Name.Trim();\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    return \"Guest\";\n"
            "}"
        ),
        "TEST_CASE_DESIGN": (
            "public static bool IsValidEmail(string email)\n"
            "{\n"
            "    if (string.IsNullOrWhiteSpace(email)) return false;\n"
            "    var at = email.IndexOf('@');\n"
            "    if (at <= 0 || at != email.LastIndexOf('@')) return false;\n"
            "    var domain = email[(at + 1)..];\n"
            "    return domain.Contains('.') && !domain.StartsWith('.') && !domain.EndsWith('.');\n"
            "}"
        ),
        "PERFORMANCE_ANALYSIS": (
            "public List<User> GetActiveUsers(List<User> users, List<Order> orders)\n"
            "{\n"
            "    var active = new List<User>();\n"
            "    foreach (var u in users)\n"
            "    {\n"
            "        foreach (var o in orders)\n"
            "        {\n"
            "            if (o.UserId == u.Id && o.Status == \"Paid\")\n"
            "            {\n"
            "                active.Add(u);\n"
            "                break;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "    return active;\n"
            "}"
        ),
        "SYSTEM_DESIGN": "",
    }
    return samples.get(template_id, "")


_GENERIC_FALLBACK_MARKERS = (
    "IOrderRepository",
    "GetCustomerOrderTotalAsync",
    "class OrderService",
)
_CODING_TASK_HINTS = (
    "hoàn thiện",
    "complete the",
    "điền vào",
    "tìm bug",
    "tìm lỗi",
    "lỗi logic",
    "fix the bug",
    "phân tích đoạn code",
    "đoạn code sau",
    "bug detection",
    "refactor",
    "viết hàm",
    "write a function",
    "debug this",
)
_GIT_TOPIC_HINTS = (
    "git",
    "pull request",
    "branching",
    "rebase",
    "merge request",
    "commit",
    "gitignore",
)


def _question_topic(q: GeneratedQuestionItem) -> str:
    return " ".join(
        str(x or "") for x in (q.question, q.skill, q.focus_area, q.rationale)
    ).lower()


def _is_generic_fallback_snippet(snippet: str | None) -> bool:
    text = snippet or ""
    return any(m in text for m in _GENERIC_FALLBACK_MARKERS)


def _is_coding_exercise(q: GeneratedQuestionItem) -> bool:
    blob = _question_topic(q)
    return any(h in blob for h in _CODING_TASK_HINTS)


def _snippet_matches_topic(q: GeneratedQuestionItem) -> bool:
    snippet = q.code_snippet or ""
    if not snippet.strip():
        return True
    if _is_generic_fallback_snippet(snippet):
        return False
    topic = _question_topic(q)
    if any(k in topic for k in _GIT_TOPIC_HINTS):
        low = snippet.lower()
        return any(
            k in low
            for k in ("git ", "git\n", "merge", "rebase", "branch", "commit", "pull request", ".gitignore")
        )
    return True


def _clear_code_fields(q: GeneratedQuestionItem) -> None:
    q.code_template_type = None
    q.code_snippet = None
    q.answer_method = "Text"


def _template_requires_snippet(template_id: str | None) -> bool:
    return (template_id or "").strip().upper() in _TEMPLATES_REQUIRING_SNIPPET


def _ensure_code_heavy_has_snippet(q: GeneratedQuestionItem) -> None:
    """Template code-heavy bắt buộc có code_snippet đề — inject default nếu thiếu."""
    tpl = (q.code_template_type or "").strip().upper()
    if not _template_requires_snippet(tpl):
        return
    if (q.code_snippet or "").strip():
        return
    logger.warning(
        "Thiếu code_snippet đề cho template=%s — inject default stem",
        tpl,
    )
    q.code_snippet = _default_snippet_for_template(tpl) or None


def _apply_outline_answer_method(q: GeneratedQuestionItem, preferred: str | None) -> None:
    """Outline HR: set answerMethod; Text không xóa đề nếu template code-heavy / đã có stem."""
    if not preferred:
        return
    q.answer_method = preferred
    if preferred == "Text":
        tpl = (q.code_template_type or "").strip().upper()
        has_stem = bool((q.code_snippet or "").strip())
        if not _template_requires_snippet(tpl) and not has_stem:
            q.code_snippet = None
            q.code_template_type = None
    elif preferred == "Code" and not (q.code_template_type or "").strip():
        pass


def _align_code_to_question(q: GeneratedQuestionItem, content_mode: str) -> None:
    """Câu Git/lý thuyết không được dán snippet C# generic (OrderService).

    Template code-heavy: giữ đề (code_snippet); Answer Text vẫn có thể có snippet.
    """
    if content_mode == "TheoryOnly":
        _clear_code_fields(q)
        return

    tpl = (q.code_template_type or "").strip().upper()
    requires = _template_requires_snippet(tpl)

    # Snippet generic / lệch chủ đề
    if q.code_snippet and not _snippet_matches_topic(q):
        if requires and _is_coding_exercise(q):
            # Bài code thật nhưng snippet fallback lệch → thay default, giữ template
            q.code_snippet = _default_snippet_for_template(tpl) or None
            return
        _clear_code_fields(q)
        return

    # Mixed + câu lý thuyết (không phải coding) + không phải template bắt buộc snippet
    if content_mode == "Mixed" and not _is_coding_exercise(q) and not requires:
        _clear_code_fields(q)
        return

    # Sai gắn template code-heavy lên câu lý thuyết không có snippet khớp → bỏ template
    if content_mode == "Mixed" and not _is_coding_exercise(q) and requires:
        if not (q.code_snippet or "").strip():
            _clear_code_fields(q)
        else:
            _ensure_code_heavy_has_snippet(q)
        return

    if requires:
        _ensure_code_heavy_has_snippet(q)


def _default_image_hint(template_id: str | None, question_type: str | None = None) -> str:
    """Gợi ý text cho HR — không AI gen ảnh."""
    tpl = (template_id or "").upper()
    hints = {
        "CODE_COMPLETION": "Tìm ảnh screenshot đoạn code TODO/incomplete hoặc whiteboard thuật toán liên quan bài toán.",
        "BUG_DETECTION": "Tìm ảnh đoạn code có lỗi (screenshot IDE) hoặc diagram minh họa bug flow.",
        "REFACTORING": "Tìm ảnh code smell trước/sau hoặc slide clean-code liên quan.",
        "TEST_CASE_DESIGN": "Tìm ảnh bảng test case / coverage matrix hoặc screenshot unit test.",
        "PERFORMANCE_ANALYSIS": "Tìm ảnh Big-O chart, profiler screenshot hoặc diagram bottleneck.",
        "SYSTEM_DESIGN": "Tìm sơ đồ kiến trúc / sequence / data-flow từ tài liệu nội bộ (Notion, Confluence, slide).",
    }
    if tpl in hints:
        return hints[tpl]
    qt = (question_type or "").lower()
    if "system" in qt or "design" in qt:
        return hints["SYSTEM_DESIGN"]
    if "behavior" in qt or "situation" in qt:
        return "Tìm ảnh situation card / whiteboard scenario minh họa tình huống phỏng vấn (nếu hữu ích)."
    return "Tìm hình ảnh hoặc diagram minh họa khái niệm chính trong câu hỏi (slide nội bộ, whiteboard, screenshot)."


def _normalize_answer_method(raw: object) -> str | None:
    """SCRUM-400: chuẩn hóa Text|Code (case-insensitive)."""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if key in ("text", "theory", "essay"):
        return "Text"
    if key in ("code", "coding", "programming"):
        return "Code"
    return None


def _infer_answer_method(template_id: str | None, snippet: str | None) -> str:
    """Fallback khi AI thiếu answer_method."""
    tpl = (template_id or "").strip().upper()
    if tpl in _TEMPLATES_REQUIRING_SNIPPET:
        return "Code"
    if (snippet or "").strip():
        return "Code"
    return "Text"


def _parse_evaluation_criteria(raw: object) -> list[RubricCriterionItem | str]:
    """SCRUM-418: parse criteria object hoặc legacy string."""
    if not isinstance(raw, list):
        return []
    out: list[RubricCriterionItem | str] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
            continue
        if isinstance(item, dict):
            try:
                crit = RubricCriterionItem.model_validate(item)
                if crit.label.strip():
                    out.append(crit)
            except Exception:
                label = str(item.get("label") or item.get("text") or "").strip()
                if label:
                    out.append(label)
    return out


# Sinh batch để tránh timeout + JSON quá dài khi totalQuestions lớn (vd 30).
_QUESTION_BATCH_SIZE = 10
_BATCH_THRESHOLD = 12

_TECHNICAL_KEYWORDS = frozenset(
    {
        "git",
        "docker",
        "kubernetes",
        "sql",
        "api",
        "rest",
        "csharp",
        "python",
        "javascript",
        "react",
        "database",
        "algorithm",
        "microservice",
    }
)


_HR_REGEN_NOTE_RE = re.compile(r"HR_REGEN_NOTE=([^;\n]+)")


def _extract_hr_regen_note(hr_note: str | None) -> str | None:
    """Lấy lưu ý HR khi regen 1 câu (Studio). BE đã đổi ';' trong lưu ý thành ','."""
    if not hr_note:
        return None
    match = _HR_REGEN_NOTE_RE.search(hr_note)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _build_plan_retrieve_query(
    plan: QuestionGenerationPlan, hr_note: str | None
) -> str:
    """SCRUM-421: embed skill/focus từ plan — không chỉ JD."""
    parts: list[str] = []
    # Lưu ý regen của HR đứng đầu query: hr_note bên dưới bị cắt 500 ký tự,
    # nếu để cuối thì chủ đề HR yêu cầu (vd. OOP) không bao giờ được dùng để tìm tài liệu.
    regen_note = _extract_hr_regen_note(hr_note)
    if regen_note:
        parts.append(regen_note)
    for cov in plan.coverage[:10]:
        if cov.skill:
            parts.append(cov.skill.strip())
        for fa in (cov.focus_areas or [])[:4]:
            if fa:
                parts.append(str(fa).strip())
    for item in plan.recommended_question_outline[:15]:
        if item.skill:
            parts.append(item.skill.strip())
        if item.focus_area:
            parts.append(item.focus_area.strip())
    if hr_note and hr_note.strip():
        parts.append(hr_note.strip()[:500])
    seen: set[str] = set()
    ordered: list[str] = []
    for p in parts:
        key = p.lower()
        if p and key not in seen:
            seen.add(key)
            ordered.append(p)
    return " | ".join(ordered)


def _plan_needs_more_system_chunks(plan: QuestionGenerationPlan) -> bool:
    query = _build_plan_retrieve_query(plan, None).lower()
    if any(kw in query for kw in _TECHNICAL_KEYWORDS):
        return True
    for cov in plan.coverage:
        for sf in cov.source_files or []:
            if sf and not is_jd_source_file(str(sf)):
                return True
    return False


def _build_skills_retrieve_query(request: GenerateQuestionsRequest) -> str | None:
    parts = [s.strip() for s in (request.skills or []) if s.strip()]
    if request.hr_note and request.hr_note.strip():
        parts.append(request.hr_note.strip()[:500])
    return " | ".join(parts) if parts else None


class QuestionGenerationService:
    def __init__(
        self,
        retrieval: RagRetrievalService,
        client: OpenAI,
        settings: Settings,
    ):
        self._retrieval = retrieval
        self._client = client
        self._settings = settings
        self._debug = settings.debug

    def _merge_system_chunks_by_skills(
        self,
        base: list[RetrievedChunk],
        plan: QuestionGenerationPlan,
    ) -> list[RetrievedChunk]:
        """Retrieve thêm theo skill unique — merge/dedupe để pool đa dạng hơn."""
        skills: list[str] = []
        seen: set[str] = set()
        for cov in plan.coverage or []:
            s = (cov.skill or "").strip()
            key = s.lower()
            if s and key not in seen:
                seen.add(key)
                skills.append(s)
        for item in plan.recommended_question_outline or []:
            s = (item.skill or item.focus_area or "").strip()
            key = s.lower()
            if s and key not in seen:
                seen.add(key)
                skills.append(s)
        if not skills:
            return base

        merged = list(base)
        keys = {
            (
                (c.metadata or {}).get("fileName", c.document_id),
                int(c.chunk_index or 0),
            )
            for c in merged
        }
        for skill in skills[:8]:
            try:
                extra = self._retrieval.retrieve_system_only(
                    skill,
                    top_k_system=3,
                    query_extra=skill,
                )
            except Exception:
                logger.warning("per-skill retrieve failed for %s", skill, exc_info=True)
                continue
            for c in extra:
                key = (
                    (c.metadata or {}).get("fileName", c.document_id),
                    int(c.chunk_index or 0),
                )
                if key in keys:
                    continue
                keys.add(key)
                merged.append(c)
        return merged

    def generate(self, request: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
        start = time.time()

        validation_error = _validate_owner_and_jd(request.owner_id, request.job_description)
        if validation_error:
            return GenerateQuestionsResponse(
                success=False,
                error=validation_error,
                processing_time_ms=(time.time() - start) * 1000,
            )

        try:
            query_extra = _build_skills_retrieve_query(request)
            top_k_system = self._settings.top_k_system
            if query_extra:
                top_k_system = max(top_k_system, 8)
            system_chunks, hr_chunks = self._retrieval.retrieve_for_job(
                request.job_description,
                request.owner_id.strip(),
                query_extra=query_extra,
                document_ids=request.document_ids or None,
                top_k_system=top_k_system,
            )
        except Exception as exc:
            logger.exception("Retrieval failed")
            return GenerateQuestionsResponse(
                success=False,
                error=str(exc),
                processing_time_ms=(time.time() - start) * 1000,
            )

        all_chunks = system_chunks + hr_chunks
        if not all_chunks:
            return GenerateQuestionsResponse(
                success=False,
                error="Không tìm thấy tài liệu liên quan trong SYSTEM hoặc HR scope.",
                processing_time_ms=(time.time() - start) * 1000,
            )

        user_message = self._build_user_message(request, system_chunks, hr_chunks)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _interview_system_prompt(request.language)},
            {"role": "user", "content": user_message},
        ]

        raw, llm_error = self._call_llm_safe(messages)
        if llm_error:
            return GenerateQuestionsResponse(
                success=False,
                error=llm_error,
                processing_time_ms=(time.time() - start) * 1000,
            )

        chunk_lookup = build_chunk_lookup(all_chunks)
        questions, parse_error = self._parse_questions(
            raw, chunk_lookup, request.job_description, request.hr_note
        )

        if parse_error and is_retryable_json_error(parse_error):
            raw, parse_error, questions = self._retry_parse(
                messages,
                raw,
                chunk_lookup,
                lambda r, lk: self._parse_questions(
                    r, lk, request.job_description, request.hr_note
                ),
            )

        questions = apply_provenance_to_questions(
            questions,
            chunks=all_chunks,
            job_description=request.job_description,
        )

        return GenerateQuestionsResponse(
            success=parse_error is None and len(questions) > 0,
            questions=questions,
            raw_answer=raw if self._debug else None,
            processing_time_ms=(time.time() - start) * 1000,
            error=parse_error,
        )

    def generate_from_plan(
        self, request: GenerateQuestionsFromPlanRequest
    ) -> GenerateQuestionsFromPlanResponse:
        start = time.time()

        validation_error = _validate_owner_and_jd(request.owner_id, request.job_description)
        if validation_error:
            return GenerateQuestionsFromPlanResponse(
                success=False,
                error=validation_error,
                processing_time_ms=(time.time() - start) * 1000,
            )

        plan_error = _validate_approved_plan(request.approved_plan)
        if plan_error:
            return GenerateQuestionsFromPlanResponse(
                success=False,
                error=plan_error,
                processing_time_ms=(time.time() - start) * 1000,
            )

        # HG01/HG02: ép type outline khớp phân bổ HR TRƯỚC khi dựng prompt — slot đổi nhóm type được
        # đổi cả skill + goal, để LLM nhận outline nhất quán ngay từ đầu (không sinh xong mới đổi nhãn).
        _reconcile_outline_types_to_distribution(
            request.approved_plan, language=request.language, rewrite_slots=True
        )

        try:
            query_extra = _build_plan_retrieve_query(
                request.approved_plan, request.hr_note
            )
            top_k_system = self._settings.top_k_system
            if _plan_needs_more_system_chunks(request.approved_plan):
                top_k_system = max(top_k_system, 8)
            doc_ids = request.document_ids or None
            system_chunks, hr_chunks = self._retrieval.retrieve_for_job(
                request.job_description,
                request.owner_id.strip(),
                query_extra=query_extra or None,
                top_k_system=top_k_system,
                document_ids=doc_ids if doc_ids else None,
            )
            # Multi-skill: bổ sung chunk System theo từng skill để tránh mọi câu dính 1 chunk
            system_chunks = self._merge_system_chunks_by_skills(
                system_chunks, request.approved_plan
            )
        except Exception as exc:
            logger.exception("Retrieval failed")
            return GenerateQuestionsFromPlanResponse(
                success=False,
                error=str(exc),
                processing_time_ms=(time.time() - start) * 1000,
            )

        all_chunks = system_chunks + hr_chunks
        if not all_chunks:
            logger.warning(
                "No retrieved chunks for question generation from plan; falling back "
                "to approved plan and JD only (owner_id=%s)",
                request.owner_id,
            )

        total = request.approved_plan.total_questions
        if total > _BATCH_THRESHOLD:
            questions, parse_error = self._generate_from_plan_batched(
                request, system_chunks, hr_chunks, all_chunks
            )
        else:
            questions, parse_error = self._generate_from_plan_single(
                request, system_chunks, hr_chunks, all_chunks
            )

        flagged: list[FlaggedQuestionItem] = []
        distribution_match: bool | None = None
        if questions:
            questions, flagged, distribution_match = self._align_flag_and_regen_from_plan(
                questions,
                request,
                system_chunks,
                hr_chunks,
                all_chunks,
            )
        else:
            questions = apply_provenance_to_questions(
                questions,
                chunks=all_chunks,
                job_description=request.job_description,
            )

        # SCRUM-495: flagged ≠ fail — vẫn success để BE lưu bộ; FE cảnh báo nhẹ qua needsReview
        return GenerateQuestionsFromPlanResponse(
            success=parse_error is None and len(questions) > 0,
            questions=questions,
            processing_time_ms=(time.time() - start) * 1000,
            error=parse_error,
            flagged_questions=flagged,
            distribution_match=distribution_match,
        )

    def _generate_from_plan_single(
        self,
        request: GenerateQuestionsFromPlanRequest,
        system_chunks: list[RetrievedChunk],
        hr_chunks: list[RetrievedChunk],
        all_chunks: list[RetrievedChunk],
        *,
        order_start: int | None = None,
        order_end: int | None = None,
        batch_count: int | None = None,
    ) -> tuple[list[GeneratedQuestionItem], str | None]:
        user_message = self._build_from_plan_user_message(
            request,
            system_chunks,
            hr_chunks,
            order_start=order_start,
            order_end=order_end,
            batch_count=batch_count,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _questions_from_plan_system_prompt(request.language)},
            {"role": "user", "content": user_message},
        ]

        # Timeout riêng: batch nhỏ vẫn cần margin; full batch lớn dùng settings cao hơn
        timeout = max(float(self._settings.request_timeout_seconds), 180.0)
        raw, llm_error = self._call_llm_safe(messages, timeout=timeout)
        if llm_error:
            return [], llm_error

        chunk_lookup = build_chunk_lookup(all_chunks)
        questions, parse_error = self._parse_questions_from_plan(
            raw,
            chunk_lookup,
            request.approved_plan,
            request.job_description,
            request.hr_note,
            expected_count=batch_count,
            language=request.language,
        )

        if parse_error and is_retryable_json_error(parse_error):
            raw, parse_error, questions = self._retry_parse(
                messages,
                raw,
                chunk_lookup,
                lambda r, lk: self._parse_questions_from_plan(
                    r,
                    lk,
                    request.approved_plan,
                    request.job_description,
                    request.hr_note,
                    expected_count=batch_count,
                    language=request.language,
                ),
                timeout=timeout,
            )

        return questions, parse_error

    def _generate_from_plan_batched(
        self,
        request: GenerateQuestionsFromPlanRequest,
        system_chunks: list[RetrievedChunk],
        hr_chunks: list[RetrievedChunk],
        all_chunks: list[RetrievedChunk],
    ) -> tuple[list[GeneratedQuestionItem], str | None]:
        """Chia totalQuestions thành batch ~10 câu — giảm timeout LLM + lỗi JSON dài."""
        total = request.approved_plan.total_questions
        merged: list[GeneratedQuestionItem] = []
        errors: list[str] = []

        for start0 in range(0, total, _QUESTION_BATCH_SIZE):
            end = min(start0 + _QUESTION_BATCH_SIZE, total)
            order_start = start0 + 1
            order_end = end
            batch_count = end - start0
            logger.info(
                "generate_from_plan batch orders %s-%s (%s questions)",
                order_start,
                order_end,
                batch_count,
            )
            batch_q, err = self._generate_from_plan_single(
                request,
                system_chunks,
                hr_chunks,
                all_chunks,
                order_start=order_start,
                order_end=order_end,
                batch_count=batch_count,
            )
            if err and not batch_q:
                errors.append(f"batch {order_start}-{order_end}: {err}")
                continue
            if err:
                logger.warning("batch %s-%s soft warning: %s", order_start, order_end, err)
            # Normalize order trong batch
            for i, q in enumerate(batch_q):
                if q.order is None:
                    q.order = order_start + i
            merged.extend(batch_q)

        if not merged:
            return [], errors[0] if errors else "Không sinh được câu hỏi nào."

        # Re-number + validate total (nới lỏng số lượng; phân bổ xử lý ở align/flag)
        merged.sort(key=lambda q: q.order if q.order is not None else 0)
        for i, q in enumerate(merged, start=1):
            q.order = i

        # SCRUM-495 harden: reconcile + khóa type theo HR distribution
        _reconcile_outline_types_to_distribution(
            request.approved_plan, language=request.language, rewrite_slots=True
        )
        self._lock_questions_to_outline(merged, request.approved_plan)
        _apply_distribution_types_when_no_outline(merged, request.approved_plan)

        validation_error = _validate_questions_against_plan(
            merged,
            request.approved_plan,
            strict_count=False,
            enforce_distribution=False,
        )
        if validation_error and len(merged) < max(1, (total * 2) // 3):
            return merged, validation_error
        if validation_error:
            # Chỉ soft-accept thiếu/thừa nhẹ tổng số — không nuốt lệch phân bổ im lặng
            logger.warning("Plan validation soft-accept (count only): %s", validation_error)
        return merged, None

    def _call_llm(self, messages: list[dict[str, str]], *, timeout: float | None = None) -> str:
        timeout_s = timeout or float(self._settings.request_timeout_seconds)
        kwargs: dict[str, Any] = {
            "model": self._settings.chat_model,
            "messages": messages,
            "temperature": self._settings.temperature,
            "stream": False,
            "timeout": timeout_s,
        }
        # json_object giúp giảm markdown/escape lẻ; một số host (cũ) không hỗ trợ
        try:
            response = self._client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"}
            )
        except Exception as first_exc:
            logger.warning(
                "LLM json_object mode failed (%s); retry without response_format",
                first_exc,
            )
            response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _call_llm_safe(
        self, messages: list[dict[str, str]], *, timeout: float | None = None
    ) -> tuple[str, str | None]:
        try:
            return self._call_llm(messages, timeout=timeout), None
        except Exception as exc:
            logger.exception("LLM call failed")
            return "", f"Lỗi LLM: {exc}"

    def _retry_parse(
        self,
        messages: list[dict[str, str]],
        raw: str,
        chunk_lookup: dict[tuple[str, str, int], RetrievedChunk],
        parse_fn: Any,
        *,
        timeout: float | None = None,
    ) -> tuple[str, str | None, list[GeneratedQuestionItem]]:
        logger.warning("LLM JSON parse failed, retry once")
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": build_json_fix_prompt(raw)},
        ]
        try:
            raw = self._call_llm(retry_messages, timeout=timeout)
        except Exception as exc:
            logger.exception("LLM retry call failed")
            return raw, f"Lỗi LLM khi retry: {exc}", []

        questions, parse_error = parse_fn(raw, chunk_lookup)
        return raw, parse_error, questions

    def _build_user_message(
        self,
        request: GenerateQuestionsRequest,
        system_chunks: list[RetrievedChunk],
        hr_chunks: list[RetrievedChunk],
    ) -> str:
        skills_text = ", ".join(request.skills) if request.skills else ""
        types_text = ", ".join(request.question_types)

        lines = [
            "[YÊU CẦU]",
            f"ownerId: {request.owner_id}",
            f"jobDescription: {request.job_description}",
            f"so_cau: {request.number_of_questions}",
            f"difficulty: {request.difficulty}",
            f"loai_cau: {types_text}",
            language_instruction_block(request.language),
        ]
        if skills_text:
            lines.append(f"skills: {skills_text}")
        if request.hr_note and request.hr_note.strip():
            lines.append(f"hrNote: {request.hr_note.strip()}")
        mode, templates = _parse_content_preferences(request.hr_note)
        lines.append(f"contentMode: {mode}")
        lines.append(f"enabledCodeTemplates: {', '.join(templates)}")
        if mode != "TheoryOnly":
            lines.append("")
            lines.append(_format_enabled_template_rules(templates))

        lines.append(
            "\nHướng dẫn (SCRUM-421 waterfall): JD = vì sao hỏi (citation HR đầu, usedFor why-asked). "
            f'Admin [HỆ THỐNG] = kiến thức kỹ thuật (origin SYSTEM, usedFor technical-body) khi có chunk. '
            "LLM = sample_answer/rubric không map chunk (origin LLM + reason). "
            f'Citation JD: source_file="{JD_SOURCE_FILE}", excerpt nguyên văn đoạn khớp skill (SCRUM-425 chunk_index theo đoạn, không luôn 0).'
        )
        lines.append("\n" + format_retrieved_context(system_chunks, hr_chunks))
        lines.append(
            f"\nTạo đúng {request.number_of_questions} câu hỏi. Trả về JSON theo schema."
        )
        return "\n".join(lines)

    def _build_from_plan_user_message(
        self,
        request: GenerateQuestionsFromPlanRequest,
        system_chunks: list[RetrievedChunk],
        hr_chunks: list[RetrievedChunk],
        *,
        order_start: int | None = None,
        order_end: int | None = None,
        batch_count: int | None = None,
    ) -> str:
        # HG01: plannedSkill/relabeled là dấu nội bộ BE — không đưa vào prompt cho LLM đỡ nhiễu
        plan_json = request.approved_plan.model_dump(
            by_alias=True,
            mode="json",
            exclude={
                "recommended_question_outline": {"__all__": {"planned_skill", "relabeled"}}
            },
        )
        target = batch_count or request.approved_plan.total_questions

        lines = [
            "[YÊU CẦU]",
            f"ownerId: {request.owner_id}",
            f"jobDescription: {request.job_description}",
            language_instruction_block(request.language),
        ]
        if request.hr_note and request.hr_note.strip():
            lines.append(f"hrNote: {request.hr_note.strip()}")
            if "STRICT_FOCUS=1" in request.hr_note:
                lines.append(
                    "STRICT_FOCUS: Chỉ sinh câu hỏi bám coverage/outline/FOCUS_AREAS. "
                    "Không hỏi skill khác dù JD còn đề cập. JD chỉ dùng cho citation why-asked."
                )
            # SCRUM-429 / SCRUM-496: Studio regen — tránh trùng; ưu tiên HR_REGEN_NOTE nếu có
            if "STUDIO_REGEN=1" in request.hr_note or "AVOID_QUESTIONS=" in request.hr_note:
                regen_note = _extract_hr_regen_note(request.hr_note)
                if regen_note:
                    # HR chủ động ghi lưu ý khi regen → lưu ý là ưu tiên cao nhất, kể cả khi
                    # nó yêu cầu chủ đề khác slot (trước đây bị cấm đổi skill nên LLM chỉ
                    # viết lại câu cũ theo dạng khác).
                    lines.append(
                        f"HR_REGEN_NOTE (ƯU TIÊN CAO NHẤT): \"{regen_note}\". "
                        "Câu mới PHẢI làm đúng lưu ý này. Nếu lưu ý yêu cầu một chủ đề/skill khác "
                        "với skill/focus_area/goal của outline (vd. outline là Middleware nhưng HR "
                        "ghi 'hỏi về OOP'), hãy BỎ chủ đề outline và hỏi đúng chủ đề HR yêu cầu; "
                        "khi đó ghi skill và focus_area MỚI đúng với câu hỏi. Nếu lưu ý chỉ đổi "
                        "góc hỏi/độ khó/format thì giữ skill/focus của outline. "
                        "Giữ type của outline và ngôn ngữ đầu ra. JD chỉ là ngữ cảnh vị trí. "
                        "AVOID_QUESTIONS: câu mới không được trùng hay paraphrase các mục trong "
                        "AVOID_QUESTIONS (phân tách |#|, mục đầu thường là câu đang regen)."
                    )
                else:
                    lines.append(
                        "STUDIO_REGEN / AVOID_QUESTIONS: Câu mới PHẢI khác ý và cấu trúc các mục "
                        "trong AVOID_QUESTIONS (phân tách |#|). Không paraphrase gần như giống; "
                        "đổi góc hỏi / ví dụ / yêu cầu cụ thể trong khi vẫn bám skill/focus/goal của outline."
                    )
                lines.append(
                    "REGEN_SAMPLE_ANSWER: sample_answer BẮT BUỘC là chuỗi không rỗng, cùng ngôn ngữ với câu hỏi "
                    "và khớp nội dung câu mới. Không được trả sample_answer rỗng."
                )
        mode, templates = _parse_content_preferences(request.hr_note)
        lines.append(f"contentMode: {mode}")
        lines.append(f"enabledCodeTemplates: {', '.join(templates)}")
        lines.append("Nếu mode=TheoryOnly: chỉ câu hỏi lý thuyết, không tạo code_snippet.")
        lines.append(
            "Nếu mode=CodeOnly: mỗi câu cần code_template_type hợp lệ trong enabledCodeTemplates; "
            "template code bắt buộc có code_snippet thật (không placeholder)."
        )
        lines.append("Nếu mode=Mixed: câu lý thuyết (Git/PR/khái niệm) = Text, không snippet; chỉ câu bài code mới có code_snippet đúng skill.")
        lines.append(
            "Mỗi câu BẮT BUỘC có image_hint (1–2 câu): gợi ý HR nên tìm/đính kèm hình hoặc diagram nào. "
            "KHÔNG yêu cầu AI gen ảnh; chỉ text gợi ý."
        )
        lines.append(
            "SCRUM-400: mỗi câu BẮT BUỘC có answer_method = Text|Code "
            "(Code = ứng viên viết/sửa code; Text = trả lời văn xuôi / SYSTEM_DESIGN)."
        )
        if mode != "TheoryOnly":
            lines.append("")
            lines.append(_format_enabled_template_rules(templates))
        lines.extend([
            "",
            "[APPROVED PLAN — KHÔNG ĐƯỢC THAY ĐỔI]",
            json.dumps(plan_json, ensure_ascii=False, indent=2),
            "",
            "Hướng dẫn (SCRUM-421 waterfall + SCRUM-426): JD = why-asked (citation HR đầu). "
            "Admin [HỆ THỐNG] = technical-body khi có chunk. "
            "LLM = phần bổ sung không map chunk (origin + reason). "
            f'JD citation: source_file="{JD_SOURCE_FILE}", excerpt nguyên văn đoạn khớp skill/focus. '
            "Nếu approvedPlan.recommendedQuestionOutline[i].citations đã khóa — "
            "BẮT BUỘC bám skill/focus/goal của slot đó; không đổi why-asked JD chunk đã khóa. "
            "rationale PHẢI trùng outline[i].goal (đã khóa ở Live Preview) — không viết lại lý do khác. "
            "HG01: mỗi câu giữ đúng order của slot và ghi skill y hệt skill của slot đó; nội dung câu "
            "phải hỏi đúng skill + goal của slot, không viết nội dung của slot khác dưới order này. "
            "Citation chunk phải khớp source_file/chunk_index trong context.",
            "Không chèn backslash lạ trong string (dùng \\\\ nếu cần ký tự \\).",
            f"Toàn bộ content câu hỏi/rationale/sample_answer/evaluation_criteria bằng "
            f"{free_text_language_label(request.language)}.",
            "",
            format_retrieved_context(system_chunks, hr_chunks),
            "",
        ])
        if order_start is not None and order_end is not None:
            lines.append(
                f"BATCH: Chỉ tạo {target} câu hỏi với order từ {order_start} đến {order_end} "
                f"(không tạo các order ngoài khoảng này). Trả về JSON theo schema."
            )
        else:
            lines.append(
                f"Tạo đúng {target} câu hỏi theo approved plan. Trả về JSON theo schema."
            )
        return "\n".join(lines)

    def _parse_questions(
        self,
        raw: str,
        chunk_lookup: dict[tuple[str, str, int], RetrievedChunk],
        job_description: str = "",
        hr_note: str | None = None,
    ) -> tuple[list[GeneratedQuestionItem], str | None]:
        data, parse_err = extract_json_object(raw)
        if parse_err or data is None:
            return [], parse_err or "LLM không trả về JSON hợp lệ."

        items = data.get("questions", data if isinstance(data, list) else [])
        if not isinstance(items, list):
            return [], "JSON thiếu mảng 'questions'."

        questions: list[GeneratedQuestionItem] = []
        content_mode, enabled_templates = _parse_content_preferences(hr_note)
        for item in items:
            if not isinstance(item, dict):
                continue
            q_text = str(item.get("question", "")).strip()
            if not q_text:
                continue

            citations = enrich_citations(
                parse_citations(item.get("citations", [])), chunk_lookup
            )
            # SCRUM-425: ensure_jd_primary sau khi có skill/focus (xem _assign_jd_primary_citations)
            rationale = str(item.get("rationale", ""))
            skill_val = str(item.get("skill") or "").strip() or None
            focus_val = str(
                item.get("focus_area") or item.get("focusArea") or ""
            ).strip() or None

            questions.append(
                GeneratedQuestionItem(
                    question=q_text,
                    question_type=str(
                        item.get("question_type", item.get("questionType", "technical"))
                    ),
                    difficulty=str(item.get("difficulty", "medium")),
                    rationale=rationale,
                    sample_answer=str(
                        item.get("sample_answer", item.get("sampleAnswer", ""))
                    ).strip(),
                    citations=citations,
                    skill=skill_val,
                    focus_area=focus_val,
                    code_template_type=str(item.get("code_template_type", item.get("codeTemplateType", ""))).strip() or None,
                    code_snippet=str(item.get("code_snippet", item.get("codeSnippet", ""))).strip() or None,
                    image_hint=str(item.get("image_hint", item.get("imageHint", ""))).strip() or None,
                    answer_method=_normalize_answer_method(
                        item.get("answer_method", item.get("answerMethod"))
                    ),
                    # HG01: skill LLM tự ghi, giữ trước khi khoá theo slot
                    llm_skill_echo=skill_val,
                )
            )

        if not questions:
            return [], "JSON không có câu hỏi hợp lệ."

        for idx, q in enumerate(questions):
            if content_mode == "TheoryOnly":
                q.code_template_type = None
                q.code_snippet = None
                q.answer_method = "Text"
            else:
                llm_template = (q.code_template_type or "").strip().upper() or None
                q.code_template_type = llm_template
                if content_mode == "CodeOnly":
                    if q.code_template_type not in enabled_templates:
                        q.code_template_type = enabled_templates[idx % len(enabled_templates)]
                elif q.code_template_type and q.code_template_type not in enabled_templates:
                    q.code_template_type = None

                q.code_snippet = _harden_code_snippet(q.code_snippet, q.code_template_type)

                needs_snippet = _template_requires_snippet(q.code_template_type)
                # Mixed + CodeOnly: template code-heavy thiếu đề → fallback default stem
                if needs_snippet and not q.code_snippet:
                    q.code_snippet = _default_snippet_for_template(q.code_template_type or "") or None

                if q.code_template_type == "SYSTEM_DESIGN" and not (q.code_snippet or "").strip():
                    q.code_snippet = None

                if content_mode == "CodeOnly":
                    q.answer_method = "Code"
                elif not q.answer_method:
                    q.answer_method = _infer_answer_method(q.code_template_type, q.code_snippet)

            _align_code_to_question(q, content_mode)
            _ensure_code_heavy_has_snippet(q)

            # Mọi mode: luôn có image_hint gợi ý cho HR
            if not (q.image_hint or "").strip():
                q.image_hint = _default_image_hint(q.code_template_type, q.question_type)

            if not q.answer_method:
                q.answer_method = _infer_answer_method(q.code_template_type, q.code_snippet)

        # SCRUM-425: gán JD chunk theo skill sau khi parse (luồng không plan)
        if job_description:
            self._assign_jd_primary_citations(questions, job_description)
        return questions, None

    @staticmethod
    def _has_locked_jd_citation(citations: list) -> bool:
        for c in citations or []:
            if is_jd_source_file(getattr(c, "source_file", None)) and (
                getattr(c, "excerpt", None) or ""
            ).strip():
                return True
        return False

    @staticmethod
    def _assign_jd_primary_citations(
        questions: list[GeneratedQuestionItem],
        job_description: str,
        *,
        skip_locked: bool = False,
    ) -> None:
        """SCRUM-425/426: chọn unit JD theo skill; skip câu đã khóa từ outline."""
        used_indexes: set[int] = set()
        for q in questions:
            if skip_locked and QuestionGenerationService._has_locked_jd_citation(
                q.citations
            ):
                for c in q.citations or []:
                    if is_jd_source_file(c.source_file) and c.chunk_index is not None:
                        used_indexes.add(int(c.chunk_index))
                continue
            jd_hint = " ".join(
                part
                for part in [
                    q.question or "",
                    q.skill or "",
                    q.focus_area or "",
                    q.rationale or "",
                ]
                if part
            )
            q.citations = ensure_jd_primary_citations(
                list(q.citations or []),
                job_description,
                hint=jd_hint,
                used_indexes=used_indexes,
            )

    @staticmethod
    def _seed_rationale_from_outline(
        questions: list[GeneratedQuestionItem],
        approved_plan: QuestionGenerationPlan,
    ) -> None:
        """SCRUM-427: rationale = outline.goal đã khóa (ghi đè LLM)."""
        outline_by_order = {
            o.order: o for o in (approved_plan.recommended_question_outline or [])
        }
        for idx, q in enumerate(questions):
            outline = None
            if q.order is not None and q.order in outline_by_order:
                outline = outline_by_order[q.order]
            elif idx < len(approved_plan.recommended_question_outline or []):
                outline = approved_plan.recommended_question_outline[idx]
            if outline is None:
                continue
            goal = (outline.goal or "").strip()
            if goal:
                q.rationale = goal

    @staticmethod
    def _seed_citations_from_outline(
        questions: list[GeneratedQuestionItem],
        approved_plan: QuestionGenerationPlan,
    ) -> None:
        """SCRUM-426: copy citations đã khóa trên outline slot vào câu hỏi."""
        outline_by_order = {
            o.order: o for o in (approved_plan.recommended_question_outline or [])
        }
        for idx, q in enumerate(questions):
            outline = None
            if q.order is not None and q.order in outline_by_order:
                outline = outline_by_order[q.order]
            elif idx < len(approved_plan.recommended_question_outline or []):
                outline = approved_plan.recommended_question_outline[idx]
            if outline is None or not outline.citations:
                continue
            if q.topic_overridden:
                # Câu đã đổi chủ đề theo lưu ý HR → citation của slot cũ không còn đúng.
                continue
            slot_cits = list(outline.citations)
            has_slot_system = any(
                (c.origin or "").upper() == "SYSTEM"
                or (
                    (c.knowledge_base or "").lower() == "system"
                    and not is_jd_source_file(c.source_file)
                )
                for c in slot_cits
            )
            extras = []
            for c in q.citations or []:
                if is_jd_source_file(c.source_file):
                    continue
                if has_slot_system and (
                    (c.origin or "").upper() == "SYSTEM"
                    or (c.knowledge_base or "").lower() == "system"
                ):
                    continue
                extras.append(c)
            q.citations = slot_cits + extras

    def _parse_questions_from_plan(
        self,
        raw: str,
        chunk_lookup: dict[tuple[str, str, int], RetrievedChunk],
        approved_plan: QuestionGenerationPlan,
        job_description: str = "",
        hr_note: str | None = None,
        *,
        expected_count: int | None = None,
        language: str | None = None,
    ) -> tuple[list[GeneratedQuestionItem], str | None]:
        questions, parse_error = self._parse_questions(
            raw, chunk_lookup, job_description, hr_note
        )
        if parse_error:
            return questions, parse_error

        # Enrich with plan-specific fields from raw JSON
        data, _ = extract_json_object(raw)
        if isinstance(data, dict):
            items = data.get("questions", [])
            if isinstance(items, list):
                # HG01: _parse_questions đã bỏ item hỏng / câu rỗng — phải lọc y hệt rồi mới ghép theo
                # index; nếu không, 1 câu rỗng là các câu sau nhận order (→ nhãn) của slot khác.
                valid_items = [
                    it
                    for it in items
                    if isinstance(it, dict) and str(it.get("question", "")).strip()
                ]
                for i, item in enumerate(valid_items):
                    if i >= len(questions):
                        continue
                    q = questions[i]
                    order_val = item.get("order")
                    if order_val is not None:
                        try:
                            q.order = int(order_val)
                        except (TypeError, ValueError):
                            pass
                    skill_val = item.get("skill")
                    if skill_val:
                        q.skill = str(skill_val)
                    focus_val = item.get("focus_area", item.get("focusArea"))
                    if focus_val:
                        q.focus_area = str(focus_val)
                    eval_raw = item.get(
                        "evaluation_criteria", item.get("evaluationCriteria", [])
                    )
                    if isinstance(eval_raw, list):
                        q.evaluation_criteria = _parse_evaluation_criteria(eval_raw)
                    tpl_val = item.get("code_template_type", item.get("codeTemplateType"))
                    if tpl_val:
                        q.code_template_type = str(tpl_val).strip()
                    snippet_val = item.get("code_snippet", item.get("codeSnippet"))
                    if snippet_val:
                        q.code_snippet = str(snippet_val)
                    hint_val = item.get("image_hint", item.get("imageHint"))
                    if hint_val:
                        q.image_hint = str(hint_val).strip()

        # SCRUM-495 harden: reconcile type outline → distribution HR, rồi khóa metadata
        _reconcile_outline_types_to_distribution(approved_plan, language=language)
        self._lock_questions_to_outline(
            questions,
            approved_plan,
            allow_topic_override=_extract_hr_regen_note(hr_note) is not None,
        )
        _apply_distribution_types_when_no_outline(questions, approved_plan)

        # SCRUM-426: copy citations đã khóa từ outline → câu hỏi
        self._seed_citations_from_outline(questions, approved_plan)

        # SCRUM-425/426: chỉ gán JD cho câu chưa có lock từ slot
        if job_description:
            self._assign_jd_primary_citations(
                questions, job_description, skip_locked=True
            )

        validation_error = _validate_questions_against_plan(
            questions,
            approved_plan,
            expected_count=expected_count,
            strict_count=expected_count is None,
            enforce_distribution=False,
        )
        mode, _ = _parse_content_preferences(hr_note)
        for q in questions:
            _align_code_to_question(q, mode)
            _ensure_code_heavy_has_snippet(q)
        return questions, validation_error

    @staticmethod
    def _lock_questions_to_outline(
        questions: list[GeneratedQuestionItem],
        approved_plan: QuestionGenerationPlan,
        *,
        allow_topic_override: bool = False,
    ) -> None:
        """SCRUM-495: metadata bám outline HR — type/skill/focus/rationale(=goal).

        allow_topic_override=True (Studio regen có HR_REGEN_NOTE): nếu LLM đã đổi sang
        skill/focus khác slot theo yêu cầu HR thì giữ skill/focus/rationale của LLM,
        chỉ khóa type + answer method.
        """
        outline_list = approved_plan.recommended_question_outline or []
        if not outline_list:
            return
        outline_by_order = {o.order: o for o in outline_list}
        for idx, q in enumerate(questions):
            outline = None
            if q.order is not None and q.order in outline_by_order:
                outline = outline_by_order[q.order]
            elif idx < len(outline_list):
                outline = outline_list[idx]
                if q.order is None:
                    q.order = outline.order
            if outline is None:
                continue
            q.question_type = _safe_normalize_question_type(outline.type)
            preferred = _normalize_answer_method(
                getattr(outline, "answer_method", None)
            )
            if allow_topic_override and _llm_changed_topic(q, outline):
                q.topic_overridden = True
                _apply_outline_answer_method(q, preferred)
                continue
            if (outline.skill or "").strip():
                q.skill = outline.skill.strip()
            if (outline.focus_area or "").strip():
                q.focus_area = outline.focus_area.strip()
            goal = (outline.goal or "").strip()
            if goal:
                q.rationale = goal
            _apply_outline_answer_method(q, preferred)

    def _align_flag_and_regen_from_plan(
        self,
        questions: list[GeneratedQuestionItem],
        request: GenerateQuestionsFromPlanRequest,
        system_chunks: list[RetrievedChunk],
        hr_chunks: list[RetrievedChunk],
        all_chunks: list[RetrievedChunk],
    ) -> tuple[list[GeneratedQuestionItem], list[FlaggedQuestionItem], bool]:
        """SCRUM-495 + HG01: reconcile distribution HR → lock theo slot → regen slot lệch → flag.

        Không còn swap nội dung giữa slot và không viết lại goal trong bộ nhớ: BE giữ mỗi slot nhất
        quán (skill + goal + nguồn), sửa ngầm ở đây chỉ làm Why ask lệch với plan đã lưu bên BE.
        """
        if not questions:
            return questions, [], True

        plan = request.approved_plan
        # HG02: ép outline.type khớp questionTypeDistribution (no-op nếu generate_from_plan đã làm)
        _reconcile_outline_types_to_distribution(
            plan, language=request.language, rewrite_slots=True
        )
        allow_topic_override = _extract_hr_regen_note(request.hr_note) is not None
        self._lock_questions_to_outline(
            questions, plan, allow_topic_override=allow_topic_override
        )
        # Không có outline: vẫn ép type theo distribution HR theo thứ tự order
        _apply_distribution_types_when_no_outline(questions, plan)

        outline_by_order = {
            o.order: o for o in (plan.recommended_question_outline or [])
        }
        expected_dist = _expected_type_counts(plan)

        # HG01: câu lệch slot (nội dung khác skill/goal, hoặc LLM ghi skill của slot khác) → regen đúng slot
        for attempt in range(_LABEL_MISMATCH_MAX_RETRIES):
            mismatched = [
                q
                for q in questions
                if q.order is not None
                # Regen có lưu ý HR: "lệch slot" có thể là đúng ý HR (đổi chủ đề) —
                # không tự regen kéo câu về chủ đề cũ; nếu thật sự lệch thì chỉ gắn cờ bên dưới.
                and not allow_topic_override
                and not q.topic_overridden
                and _slot_mismatch_reasons(q, outline_by_order.get(q.order), outline_by_order)
            ]
            if not mismatched:
                break
            logger.info(
                "SCRUM-495 label mismatch regen round=%s orders=%s",
                attempt + 1,
                [q.order for q in mismatched],
            )
            for q in mismatched:
                order = q.order
                if order is None:
                    continue
                replacement, err = self._generate_from_plan_single(
                    request,
                    system_chunks,
                    hr_chunks,
                    all_chunks,
                    order_start=order,
                    order_end=order,
                    batch_count=1,
                )
                if err or not replacement:
                    logger.warning(
                        "SCRUM-495 regen slot %s failed: %s", order, err
                    )
                    continue
                new_q = replacement[0]
                new_q.order = order
                self._lock_questions_to_outline(
                    [new_q], plan, allow_topic_override=allow_topic_override
                )
                _apply_distribution_types_when_no_outline([new_q], plan)
                idx = next(
                    (i for i, x in enumerate(questions) if x.order == order),
                    None,
                )
                if idx is not None:
                    questions[idx] = new_q

        # HG02: vẫn lệch count → đổi type câu thừa sang loại thiếu + regen
        for round_i in range(_DISTRIBUTION_FIX_MAX_ROUNDS):
            actual_dist = Counter(
                _safe_normalize_question_type(q.question_type) for q in questions
            )
            if not expected_dist or actual_dist == expected_dist:
                break
            to_fix = _force_types_toward_distribution(questions, expected_dist)
            if not to_fix:
                break
            # Cập nhật outline.type cho các order vừa ép (để lock/regen bám đúng)
            for q in to_fix:
                if q.order is not None and q.order in outline_by_order:
                    outline_by_order[q.order].type = _safe_normalize_question_type(
                        q.question_type
                    )
            logger.info(
                "SCRUM-495 distribution force+regen round=%s orders=%s expected=%s actual=%s",
                round_i + 1,
                [q.order for q in to_fix],
                dict(expected_dist),
                dict(actual_dist),
            )
            for q in to_fix:
                order = q.order
                if order is None:
                    continue
                replacement, err = self._generate_from_plan_single(
                    request,
                    system_chunks,
                    hr_chunks,
                    all_chunks,
                    order_start=order,
                    order_end=order,
                    batch_count=1,
                )
                if err or not replacement:
                    continue
                new_q = replacement[0]
                new_q.order = order
                # Giữ type đã ép theo distribution
                forced = _safe_normalize_question_type(q.question_type)
                self._lock_questions_to_outline(
                    [new_q], plan, allow_topic_override=allow_topic_override
                )
                new_q.question_type = forced
                idx = next(
                    (i for i, x in enumerate(questions) if x.order == order),
                    None,
                )
                if idx is not None:
                    questions[idx] = new_q

        # HG01: KHÔNG rewrite Domain/Why ask theo content — giữ metadata slot HR
        # Flag theo content ↔ outline slot (sau regen vẫn lệch)
        flagged: list[FlaggedQuestionItem] = []
        for q in questions:
            q.needs_review = False
            q.mismatch_reasons = []
            reasons: list[str] = []
            outline = outline_by_order.get(q.order) if q.order is not None else None
            if not q.topic_overridden:
                reasons.extend(_slot_mismatch_reasons(q, outline, outline_by_order))
            if outline is not None:
                want = _safe_normalize_question_type(outline.type)
                got = _safe_normalize_question_type(q.question_type)
                if got != want:
                    reasons.append(f"type_mismatch:expected={want}:actual={got}")
            if reasons:
                q.needs_review = True
                q.mismatch_reasons = list(dict.fromkeys(reasons))
                flagged.append(
                    FlaggedQuestionItem(order=q.order or 0, reasons=q.mismatch_reasons)
                )

        actual_final = Counter(
            _safe_normalize_question_type(q.question_type) for q in questions
        )
        distribution_match = (not expected_dist) or actual_final == expected_dist
        if not distribution_match:
            logger.warning(
                "SCRUM-495 distribution still mismatched: expected=%s actual=%s",
                dict(expected_dist),
                dict(actual_final),
            )
            reason = (
                f"distribution_mismatch:expected={dict(expected_dist)}"
                f":actual={dict(actual_final)}"
            )
            for q in questions:
                if not q.needs_review:
                    q.needs_review = True
                    q.mismatch_reasons = [reason]
                    flagged.append(
                        FlaggedQuestionItem(order=q.order or 0, reasons=[reason])
                    )
                else:
                    q.mismatch_reasons = list(
                        dict.fromkeys([*(q.mismatch_reasons or []), reason])
                    )

        questions = apply_provenance_to_questions(
            questions,
            chunks=all_chunks,
            job_description=request.job_description,
        )
        return questions, flagged, distribution_match


def _llm_changed_topic(q: GeneratedQuestionItem, outline: Any) -> bool:
    """True khi LLM ghi skill/focus khác slot (chỉ dùng lúc regen có lưu ý HR)."""
    if q.topic_overridden:
        return True
    llm_skill = (q.skill or "").strip().lower()
    llm_focus = (q.focus_area or "").strip().lower()
    slot_skill = (getattr(outline, "skill", None) or "").strip().lower()
    slot_focus = (getattr(outline, "focus_area", None) or "").strip().lower()
    if llm_skill and llm_skill != slot_skill:
        return True
    return bool(llm_focus) and llm_focus != slot_focus


def _safe_normalize_question_type(value: str | None) -> str:
    raw = (value or "technical").strip()
    try:
        return normalize_question_type(raw)
    except ValueError:
        key = raw.lower().replace(" ", "-").replace("_", "-")
        aliases = {
            "culture": "situational",
            "culture-fit": "situational",
            "culturefit": "situational",
            "systemdesign": "system-design",
            "problemsolving": "problem-solving",
            "behavioural": "behavioral",
            "follow-up": "technical",
            "followup": "technical",
        }
        key = aliases.get(key, key)
        return key if key in VALID_QUESTION_TYPES else "technical"


def _expected_type_counts(plan: QuestionGenerationPlan) -> Counter:
    """SCRUM-495 harden: ưu tiên questionTypeDistribution (config HR UI)."""
    dist_items = plan.question_type_distribution or []
    if dist_items:
        return Counter(
            {
                _safe_normalize_question_type(getattr(item, "type", "technical")): int(
                    item.count
                )
                for item in dist_items
                if int(getattr(item, "count", 0) or 0) > 0
            }
        )
    outline = plan.recommended_question_outline or []
    if outline:
        return Counter(_safe_normalize_question_type(o.type) for o in outline)
    return Counter()


_TECHNICAL_OUTLINE_TYPES = frozenset({"technical", "system-design", "problem-solving"})
_SOFT_SKILL_LABELS = {"behavioral": "Behavioral", "situational": "Situational"}


def _canon_skill(name: str | None) -> str:
    """"React.js" / "ReactJS" / "react" → "react" (giống CanonSkill phía BE)."""
    compact = re.sub(r"[\s._\-]", "", (name or "").lower())
    return compact[:-2] if len(compact) > 4 and compact.endswith("js") else compact


def _is_technical_outline_type(normalized_type: str) -> bool:
    return normalized_type in _TECHNICAL_OUTLINE_TYPES


def _default_goal_for_slot(skill: str, slot_type: str, language: str | None) -> str:
    """Why ask mặc định khi slot đổi type/skill — câu chữ kỹ thuật giống BE (StudioOutlineSlotHelper)."""
    english = normalize_output_language(language) == "English"
    if slot_type == "behavioral":
        return (
            "Assess teamwork and communication through a real past experience."
            if english
            else "Đánh giá kỹ năng làm việc nhóm và giao tiếp qua trải nghiệm thực tế."
        )
    if slot_type == "situational":
        return (
            "Assess judgment when handling a realistic work situation."
            if english
            else "Đánh giá cách xử lý một tình huống công việc thực tế."
        )
    return (
        f"Assess hands-on {skill} knowledge relevant to this role."
        if english
        else f"Đánh giá kiến thức {skill} thực tế cho vị trí này."
    )


def _least_covered_technical_skill(plan: QuestionGenerationPlan) -> str:
    """Slot mềm chuyển thành kỹ thuật: chọn skill của plan đang có ít slot kỹ thuật nhất."""
    candidates = [c.skill for c in (plan.coverage or []) if (c.skill or "").strip()]
    if not candidates:
        candidates = [s for s in (plan.skills or []) if (s or "").strip()]
    if not candidates:
        return "Technical"
    used = Counter(
        _canon_skill(o.skill)
        for o in (plan.recommended_question_outline or [])
        if _is_technical_outline_type(_safe_normalize_question_type(o.type))
    )
    return min(candidates, key=lambda s: used.get(_canon_skill(s), 0))


def _rewrite_slot_for_new_type(
    slot: Any,
    old_type: str,
    new_type: str,
    plan: QuestionGenerationPlan,
    language: str | None,
) -> None:
    """HG01: slot đổi nhóm kỹ thuật ↔ mềm thì đổi luôn skill + goal và bỏ nguồn cũ của skill trước."""
    was_technical = _is_technical_outline_type(old_type)
    is_technical = _is_technical_outline_type(new_type)
    if was_technical == is_technical:
        return
    if is_technical:
        skill = _least_covered_technical_skill(plan)
    else:
        skill = _SOFT_SKILL_LABELS.get(new_type, new_type.title())
    slot.skill = skill
    slot.focus_area = skill
    slot.goal = _default_goal_for_slot(skill, new_type, language)
    slot.citations = []
    slot.planned_skill = skill
    slot.relabeled = True


def _reconcile_outline_types_to_distribution(
    plan: QuestionGenerationPlan,
    *,
    language: str | None = None,
    rewrite_slots: bool = False,
) -> int:
    """
    Nếu outline đếm type ≠ distribution HR → sửa outline.type in-memory.
    Trả về số slot đã đổi type.
    """
    expected = _expected_type_counts(plan)
    outline = plan.recommended_question_outline or []
    if not expected or not outline:
        return 0

    actual = Counter(_safe_normalize_question_type(o.type) for o in outline)
    if actual == expected:
        return 0

    changed = 0
    # Xây hàng đợi loại còn thiếu
    deficit: list[str] = []
    for t, need in expected.items():
        have = actual.get(t, 0)
        if need > have:
            deficit.extend([t] * (need - have))

    # Slot thừa (loại vượt expected) — ưu tiên order lớn
    surplus_slots = []
    running = Counter()
    for o in sorted(outline, key=lambda x: x.order):
        t = _safe_normalize_question_type(o.type)
        running[t] += 1
        if running[t] > expected.get(t, 0):
            surplus_slots.append(o)

    for o in surplus_slots:
        if not deficit:
            break
        new_t = deficit.pop(0)
        old = _safe_normalize_question_type(o.type)
        if old != new_t:
            logger.info(
                "SCRUM-495 reconcile outline order=%s type %s → %s (match HR distribution)",
                o.order,
                old,
                new_t,
            )
            o.type = new_t
            # HG01: luồng HR — đổi type là đổi cả slot, không để slot behavioral mang skill/goal kỹ thuật.
            # Luồng Coach/Candidate giữ skill blueprint (chấm năng lực theo skill) nên không bật.
            if rewrite_slots:
                _rewrite_slot_for_new_type(o, old, new_t, plan, language)
            changed += 1

    return changed


def _apply_distribution_types_when_no_outline(
    questions: list[GeneratedQuestionItem],
    plan: QuestionGenerationPlan,
) -> None:
    """Không có outline: gán type theo distribution HR theo thứ tự câu."""
    if plan.recommended_question_outline:
        return
    expected = _expected_type_counts(plan)
    if not expected:
        return
    type_queue: list[str] = []
    # Ổn định thứ tự: technical → behavioral → situational → còn lại
    preferred = ["technical", "behavioral", "situational", "system-design", "problem-solving"]
    for t in preferred:
        if t in expected:
            type_queue.extend([t] * expected[t])
    for t, n in expected.items():
        if t not in preferred:
            type_queue.extend([t] * n)
    for i, q in enumerate(sorted(questions, key=lambda x: x.order or 0)):
        if i < len(type_queue):
            q.question_type = type_queue[i]


def _force_types_toward_distribution(
    questions: list[GeneratedQuestionItem],
    expected: Counter,
) -> list[GeneratedQuestionItem]:
    """Đổi type câu thừa → loại thiếu. Trả list câu đã đổi (cần regen)."""
    if not expected:
        return []
    actual = Counter(_safe_normalize_question_type(q.question_type) for q in questions)
    if actual == expected:
        return []

    deficit: list[str] = []
    for t, need in expected.items():
        have = actual.get(t, 0)
        if need > have:
            deficit.extend([t] * (need - have))

    changed: list[GeneratedQuestionItem] = []
    running = Counter()
    for q in sorted(questions, key=lambda x: x.order or 0):
        t = _safe_normalize_question_type(q.question_type)
        running[t] += 1
        if running[t] > expected.get(t, 0) and deficit:
            new_t = deficit.pop(0)
            q.question_type = new_t
            changed.append(q)
    return changed


def _label_tokens(text: str) -> list[str]:
    raw = (text or "").lower()
    raw = raw.replace("c#", "csharp").replace(".net", "dotnet").replace("node.js", "nodejs")
    raw = raw.replace("next.js", "nextjs").replace("asp.net", "aspnet")
    raw = raw.replace("postgresql", "postgres").replace("react.js", "react")
    parts = re.findall(r"[a-z0-9+]{2,}", raw)
    stop = {
        "and", "or", "the", "for", "with", "from", "that", "this", "into", "about",
        "các", "của", "và", "trong", "khi", "một", "câu", "hỏi", "đánh", "giá",
    }
    return [p for p in parts if p not in stop and len(p) >= 2]


# Topic kỹ thuật mịn hơn domain rộng — React hooks ≠ Next.js SSR dù cùng “frontend”
# HG01: bỏ các từ tiếng Anh thông dụng ("explain", "index", "query", "join", "select", "transaction",
# "rest", "express", "compose", "interface", "lambda") — trước đây câu "Explain the useEffect hook"
# bị xếp vào SQL, "the rest of" thành REST API → regen / gắn cờ nhầm câu đúng.
_TOPIC_MARKERS: dict[str, tuple[str, ...]] = {
    "sql": (
        "postgres", "postgresql", "mysql", "sql", "sql server", "t-sql", "sqlite",
        "inner join", "left join", "join clause", "group by", "groupby",
        "explain analyze", "query plan", "execution plan", "truy vấn", "foreign key",
        "database index", "composite index", "stored procedure", "n+1 query",
    ),
    "react_hooks": (
        "hooks", "usestate", "useeffect", "usememo", "usecallback", "usecontext",
        "component-based", "component based", "kiến trúc component",
    ),
    "next_rendering": (
        "ssr", "ssg", "server-side rendering", "static site generation",
        "getserversideprops", "getstaticprops", "app router", "pages router",
    ),
    "nextjs": ("next.js", "nextjs"),
    "react": ("react.js", "reactjs", "react"),
    "rest_api": (
        "rest api", "restful", "http method", "endpoint", "status code",
        "get/post", "put/patch", "api design", "openapi", "swagger",
    ),
    "nodejs": ("node.js", "nodejs", "express.js", "expressjs", "event loop", "npm"),
    "container": (
        "docker", "kubernetes", "k8s", "container", "dockerfile", "docker compose",
        "docker-compose", "pod", "containerization",
    ),
    "dotnet": ("asp.net", "aspnet", "csharp", "c#", "dotnet", ".net", "entity framework"),
    "typescript": ("typescript", "type annotation", "kiểu dữ liệu"),
    "cloud": ("aws", "azure", "gcp", "s3", "aws lambda"),
}

# Thuật ngữ đặc trưng trong Why ask — nếu có trong rationale mà không có trong câu hỏi → lệch
_DISTINCTIVE_WHY_TERMS: tuple[str, ...] = (
    "hooks", "usestate", "useeffect", "usememo", "component-based",
    "ssr", "ssg", "getserversideprops", "getstaticprops",
    "nodejs", "node.js", "event loop",
    "docker", "kubernetes",
    "postgres", "postgresql", "mysql", "group by",
    "rest api", "restful", "graphql",
    "typescript", "kiểu dữ liệu",
    "react hooks", "react hook",
)


def _marker_in_text(hay: str, marker: str) -> bool:
    """
    HG01: marker chỉ gồm chữ/số/khoảng trắng → khớp theo ranh giới từ, cho phép số nhiều "s"
    ("rest" không khớp "interest", "endpoint" vẫn khớp "endpoints"). Marker có ký tự đặc biệt
    (node.js, c#, t-sql, tiếng Việt có dấu) → so chuỗi con như cũ.
    """
    m = marker.strip().lower()
    if not m:
        return False
    if re.fullmatch(r"[a-z0-9 ]+", m):
        return re.search(rf"(?<![a-z0-9]){re.escape(m)}s?(?![a-z0-9])", hay) is not None
    return m in hay


def _topics_in_text(text: str) -> set[str]:
    hay = f" {(text or '').lower()} "
    found: set[str] = set()
    for topic, markers in _TOPIC_MARKERS.items():
        if any(_marker_in_text(hay, m) for m in markers):
            found.add(topic)
    return found


def _domains_in_text(text: str) -> set[str]:
    """Map topic → nhóm rộng (tương thích chỗ gọi cũ)."""
    topics = _topics_in_text(text)
    wide: set[str] = set()
    mapping = {
        "sql": "sql",
        "react_hooks": "frontend",
        "next_rendering": "frontend",
        "nextjs": "frontend",
        "react": "frontend",
        "typescript": "frontend",
        "rest_api": "backend",
        "nodejs": "backend",
        "dotnet": "backend",
        "container": "container",
        "cloud": "cloud",
    }
    for t in topics:
        wide.add(mapping.get(t, t))
    return wide


def _distinctive_terms_in_text(text: str) -> set[str]:
    hay = f" {(text or '').lower()} "
    found: set[str] = set()
    for term in _DISTINCTIVE_WHY_TERMS:
        if _marker_in_text(hay, term):
            found.add(term)
    return found


def _term_in_content(content_hay: str, term: str) -> bool:
    """Why ask ghi "hooks", câu hỏi ghi "hook" vẫn là cùng ý — so cả dạng số ít."""
    singular = term[:-1] if term.endswith("s") and len(term) > 3 else term
    return _marker_in_text(content_hay, term) or _marker_in_text(content_hay, singular)


def _label_mismatches_content(content: str, label: str) -> bool:
    """
    True nếu skill/Why ask không bám câu hỏi:
    - topic kỹ thuật xung đột, hoặc
    - Why ask nêu thuật ngữ đặc trưng (hooks, SSR, Node…) không có trong câu hỏi.
    """
    if not (content or "").strip() or not (label or "").strip():
        return False

    # 1) Topic conflict (React hooks vs Next SSR; REST vs React; SQL vs container…)
    ct = _topics_in_text(content)
    lt = _topics_in_text(label)
    if ct and lt and not (ct & lt):
        # Ngoại lệ: nextjs + next_rendering cùng họ; react + react_hooks
        related = [
            {"nextjs", "next_rendering"},
            {"react", "react_hooks"},
            {"nodejs", "rest_api"},
            {"dotnet", "rest_api"},
        ]
        loosely_ok = any((ct & pair) and (lt & pair) for pair in related)
        if not loosely_ok:
            return True

    # 2) Distinctive terms in label missing from content (hooks trong Why ask, câu hỏi chỉ SSR)
    label_terms = _distinctive_terms_in_text(label)
    if not label_terms:
        return False
    content_hay = f" {(content or '').lower()} "
    if not any(_term_in_content(content_hay, t) for t in label_terms):
        return True
    return False


def _domains_conflict(content: str, label: str) -> bool:
    """True khi content và Why ask/skill lệch chủ đề (topic hoặc thuật ngữ đặc trưng)."""
    return _label_mismatches_content(content, label)


def _infer_skill_label_from_content(question: str) -> str | None:
    hay = (question or "").lower()
    if "postgres" in hay or "postgresql" in hay:
        return "PostgreSQL"
    if "mysql" in hay:
        return "MySQL"
    if re.search(r"\b(sql|join|group by)\b", hay):
        return "SQL"
    if "docker" in hay or "container" in hay:
        return "Docker"
    if "kubernetes" in hay or "k8s" in hay:
        return "Kubernetes"
    if "ssr" in hay or "ssg" in hay or "next.js" in hay or "nextjs" in hay:
        return "Next.js"
    if "rest" in hay or "restful" in hay or "endpoint" in hay:
        return "REST API"
    if "hook" in hay or "usestate" in hay or "useeffect" in hay:
        return "React"
    if "react" in hay:
        return "React"
    if "typescript" in hay:
        return "TypeScript"
    if "node.js" in hay or "nodejs" in hay:
        return "Node.js"
    if "asp.net" in hay or "c#" in hay:
        return "ASP.NET Core"
    return None


def _build_why_ask_for_content(skill: str, focus: str, question: str) -> str:
    """Why ask ngắn khớp nội dung câu — không giữ goal lệch chủ đề."""
    sk = (skill or "").strip() or _infer_skill_label_from_content(question) or "kỹ năng liên quan"
    fo = (focus or "").strip()
    tip = re.sub(r"\s+", " ", (question or "").strip())
    if len(tip) > 100:
        tip = tip[:97] + "…"
    if fo and fo.lower() != sk.lower() and not _label_mismatches_content(question, fo):
        return f"Đánh giá {sk} ({fo}) — {tip}"
    return f"Đánh giá {sk} — {tip}"


def _topics_related(a: set[str], b: set[str]) -> bool:
    """Hai tập topic có cùng họ lỏng (next↔ssr, react↔hooks, node↔rest…)."""
    if a & b:
        return True
    related = [
        {"nextjs", "next_rendering"},
        {"react", "react_hooks"},
        {"nodejs", "rest_api"},
        {"dotnet", "rest_api"},
    ]
    return any((a & pair) and (b & pair) for pair in related)


def _rest_or_backend_dominates_frontend_slot(content: str, skill_topics: set[str]) -> bool:
    """HR30: slot React/Next nhưng câu chủ yếu REST/API/SQL/container."""
    frontend = {"react", "react_hooks", "nextjs", "next_rendering", "typescript"}
    if not (skill_topics & frontend):
        return False
    ct = _topics_in_text(content)
    foreign = ct & {"rest_api", "sql", "container", "nodejs", "dotnet", "cloud"}
    if not foreign:
        return False
    # Có nhắc React sơ sài nhưng trọng tâm REST → vẫn lệch slot
    if "rest_api" in foreign:
        hay = (content or "").lower()
        rest_hits = sum(
            1
            for m in ("rest", "api", "http", "endpoint", "status code", "method")
            if m in hay
        )
        react_hits = sum(
            1
            for m in ("hooks", "usestate", "useeffect", "jsx", "virtual dom", "component")
            if m in hay
        )
        if rest_hits >= 2 and rest_hits > react_hits:
            return True
    # foreign khác frontend và không giao frontend topic trong content
    if foreign and not (ct & frontend):
        return True
    if foreign and (ct & frontend) and len(foreign) >= 1:
        # content vừa có react vừa sql/docker → ưu tiên foreign nếu skill chỉ frontend
        if foreign & {"sql", "container", "cloud"}:
            return True
    return False


def _content_mismatches_slot(
    q: GeneratedQuestionItem,
    outline: Any | None,
) -> list[str]:
    """
    SCRUM-495 HG01: nội dung câu phải khớp skill (+ goal) của slot HR.
    Không dùng để rewrite nhãn — chỉ quyết định regen/flag.
    """
    if outline is None:
        return []
    reasons: list[str] = []
    content = f"{q.question or ''} {q.code_snippet or ''}"
    haystack = content.lower()
    skill = (getattr(outline, "skill", None) or "").strip()
    goal = (getattr(outline, "goal", None) or "").strip()

    qt = _safe_normalize_question_type(getattr(outline, "type", None))
    soft_types = {"behavioral", "situational"}
    if qt in soft_types:
        soft_markers = (
            "bạn ", "you ", "team", "conflict", "deadline", "stakeholder",
            "tinh huong", "tình huống", "ứng xử", "hanh vi", "hành vi",
            "pressure", "priority", "communicate", "collaborate",
            "xung đột", "đồng nghiệp", "sếp ", "manager",
        )
        if any(m in haystack for m in soft_markers):
            return []

    st = _topics_in_text(skill)
    ct = _topics_in_text(content)
    gt = _topics_in_text(goal)

    if skill and _rest_or_backend_dominates_frontend_slot(content, st):
        reasons.append(f"content_mismatch_skill:{skill}")
    elif st and ct and not _topics_related(st, ct):
        reasons.append(f"content_mismatch_skill:{skill}")
    elif skill:
        # Skill ngoài từ điển topic: câu nhắc skill HOẶC focus của slot đều tính là đúng slot
        # (vd slot C# / focus DI hỏi "DI là gì") — chỉ xét skill làm regen nhầm cả câu hợp lệ.
        focus = (getattr(outline, "focus_area", None) or "").strip()
        tokens = _label_tokens(skill) + _label_tokens(focus)
        compact = re.sub(r"[^a-z0-9+]", "", haystack)
        if tokens and not any(t in haystack or t in compact for t in tokens):
            if not (st and ct and _topics_related(st, ct)):
                reasons.append(f"content_mismatch_skill:{skill}")

    # Goal/Why ask (đã khóa từ outline): content phải liên quan goal nếu goal có topic rõ
    if goal and gt and ct and not _topics_related(gt, ct):
        reasons.append("content_mismatch_rationale_goal")
    elif goal and _label_mismatches_content(content, goal):
        # hooks trong goal, câu hỏi SSR — hoặc Node trong goal, câu REST trên slot React
        reasons.append("content_mismatch_rationale_goal")

    return list(dict.fromkeys(reasons))


def sanitize_outline_skill_goal(outline: list[Any], language: str | None = None) -> int:
    """
    Plan-time: nếu goal lệch topic so với skill → rewrite goal ngắn theo skill.
    Trả về số slot đã sửa.
    """
    changed = 0
    for item in outline or []:
        skill = (getattr(item, "skill", None) or "").strip()
        goal = (getattr(item, "goal", None) or "").strip()
        if not skill or not goal:
            continue
        st = _topics_in_text(skill)
        gt = _topics_in_text(goal)
        if not st or not gt:
            continue
        if _topics_related(st, gt):
            continue
        new_goal = (
            f"Assess hands-on {skill} knowledge relevant to this role."
            if language and normalize_output_language(language) == "English"
            else f"Đánh giá năng lực {skill}"
        )
        logger.warning(
            "SCRUM-495 sanitize outline order=%s goal lệch skill=%r — rewrite goal",
            getattr(item, "order", "?"),
            skill,
        )
        item.goal = new_goal
        changed += 1
    return changed


def _align_why_ask_and_skill_to_content(
    questions: list[GeneratedQuestionItem],
    outline_by_order: dict[int, Any],
) -> int:
    """DEPRECATED — không gọi trong pipeline (che lệch bằng nhãn). Giữ stub = 0."""
    _ = questions, outline_by_order
    return 0


def _score_content_to_slot(q: GeneratedQuestionItem, outline: Any) -> float:
    """Điểm khớp nội dung với skill/goal slot (cao hơn = tốt hơn)."""
    haystack = f"{q.question or ''} {q.code_snippet or ''}".lower()
    compact = re.sub(r"[^a-z0-9+]", "", haystack)
    skill = (getattr(outline, "skill", None) or "").strip()
    goal = (getattr(outline, "goal", None) or "").strip()
    score = 0.0
    for t in _label_tokens(skill):
        if t in haystack or t in compact:
            score += 3.0
    for t in _label_tokens(goal):
        if len(t) >= 4 and (t in haystack or t in compact):
            score += 1.0
    return score


def _swap_question_content(a: GeneratedQuestionItem, b: GeneratedQuestionItem) -> None:
    """Đổi nội dung giữa 2 câu — giữ order và metadata đã khóa."""
    fields = (
        "question",
        "sample_answer",
        "code_snippet",
        "code_template_type",
        "image_hint",
        "evaluation_criteria",
        "citations",
        "answer_method",
    )
    for f in fields:
        va = getattr(a, f, None)
        vb = getattr(b, f, None)
        setattr(a, f, vb)
        setattr(b, f, va)


def _reassign_mismatched_contents(
    questions: list[GeneratedQuestionItem],
    outline_by_order: dict[int, Any],
) -> int:
    """
    Greedy: với các câu lệch skill/goal, hoán nội dung 1-1 nếu content A khớp slot B hơn.
    Trả về số lần swap.
    """
    if not outline_by_order or len(questions) < 2:
        return 0

    mismatched = [
        q
        for q in questions
        if q.order is not None
        and q.order in outline_by_order
        and _content_mismatch_reasons(q, outline_by_order[q.order])
    ]
    if len(mismatched) < 2:
        return 0

    swaps = 0
    # Lặp đến khi không cải thiện
    improved = True
    while improved:
        improved = False
        for i, qi in enumerate(mismatched):
            oi = outline_by_order.get(qi.order)  # type: ignore[arg-type]
            if oi is None:
                continue
            best_j = None
            best_gain = 0.0
            score_ii = _score_content_to_slot(qi, oi)
            for j, qj in enumerate(mismatched):
                if i >= j:
                    continue
                oj = outline_by_order.get(qj.order)  # type: ignore[arg-type]
                if oj is None:
                    continue
                score_jj = _score_content_to_slot(qj, oj)
                # Sau swap: qi content ↔ qj content
                # score nếu qi nhận content qj cho slot oi, qj nhận content qi cho slot oj
                # Dùng tạm swap thử
                _swap_question_content(qi, qj)
                new_ii = _score_content_to_slot(qi, oi)
                new_jj = _score_content_to_slot(qj, oj)
                gain = (new_ii + new_jj) - (score_ii + score_jj)
                # Đảo lại để thử cặp khác
                _swap_question_content(qi, qj)
                if gain > best_gain + 0.5:
                    best_gain = gain
                    best_j = j
            if best_j is not None:
                qj = mismatched[best_j]
                _swap_question_content(qi, qj)
                swaps += 1
                improved = True
                logger.info(
                    "SCRUM-495 swapped content orders %s ↔ %s (gain=%.1f)",
                    qi.order,
                    qj.order,
                    best_gain,
                )
                break  # restart while sau mỗi swap
    return swaps


def _llm_echo_mismatch_reason(
    q: GeneratedQuestionItem,
    outline: Any | None,
    outline_by_order: dict[int, Any],
) -> str | None:
    """
    HG01: LLM tự ghi skill của câu = skill của MỘT SLOT KHÁC → nó đã viết nội dung slot khác dưới
    order này, lock sẽ dán nhãn sai. Chỉ bắt khi trùng skill slot khác để không báo nhầm kiểu
    "React" vs "React.js".
    """
    echo = _canon_skill(getattr(q, "llm_skill_echo", None))
    if not echo or outline is None:
        return None
    own = _canon_skill(getattr(outline, "skill", None))
    if not own or echo == own:
        return None
    other_skills = {
        _canon_skill(getattr(o, "skill", None))
        for order, o in outline_by_order.items()
        if order != q.order
    }
    if echo in other_skills:
        return f"llm_skill_echo_mismatch:{q.llm_skill_echo}"
    return None


def _slot_mismatch_reasons(
    q: GeneratedQuestionItem,
    outline: Any | None,
    outline_by_order: dict[int, Any],
) -> list[str]:
    """Lý do câu không khớp slot HR: nội dung ↔ skill/goal, và LLM ghi skill của slot khác."""
    reasons = list(_content_mismatch_reasons(q, outline))
    echo_reason = _llm_echo_mismatch_reason(q, outline, outline_by_order)
    if echo_reason:
        reasons.append(echo_reason)
    return list(dict.fromkeys(reasons))


def _content_mismatch_reasons(
    q: GeneratedQuestionItem,
    outline: Any | None,
) -> list[str]:
    """Trả list lý do nếu nội dung không bám skill/goal của slot HR — nguồn regen/flag."""
    return _content_mismatches_slot(q, outline)


def _validate_owner_and_jd(owner_id: str, job_description: str) -> str | None:
    if not owner_id or not owner_id.strip():
        return "ownerId là bắt buộc"
    if not job_description or not job_description.strip():
        return "jobDescription là bắt buộc"
    return None


def _validate_approved_plan(plan: QuestionGenerationPlan) -> str | None:
    if plan.total_questions < 1:
        return "approvedPlan.totalQuestions phải >= 1"
    if not plan.role_title.strip():
        return "approvedPlan.roleTitle là bắt buộc"
    return None


def _validate_questions_against_plan(
    questions: list[GeneratedQuestionItem],
    plan: QuestionGenerationPlan,
    *,
    expected_count: int | None = None,
    strict_count: bool = True,
    enforce_distribution: bool = True,
) -> str | None:
    target = expected_count if expected_count is not None else plan.total_questions
    got = len(questions)
    if got > target:
        del questions[target:]
        logger.warning(
            "trimmed extra questions: got=%s expected=%s", got, target
        )
        for i, q in enumerate(questions, start=1):
            q.order = i
        got = target
    if got != target:
        # Full generation: chấp nhận thiếu/thừa nhẹ (±2 hoặc ≥ 2/3) để tránh fail cả job
        if not strict_count:
            if got == 0:
                return "Không có câu hỏi hợp lệ."
            if got < max(1, (target * 2) // 3):
                return (
                    f"Số câu hỏi ({got}) quá thấp so với yêu cầu ({target})."
                )
            logger.warning(
                "question count soft mismatch: got=%s expected=%s", got, target
            )
            return None
        # Single shot strict-ish: nới ±2 nếu plan lớn
        if plan.total_questions >= 10 and abs(got - target) <= 2 and got > 0:
            logger.warning(
                "question count soft mismatch (±2): got=%s expected=%s", got, target
            )
            return None
        return (
            f"Số câu hỏi ({got}) "
            f"không khớp yêu cầu ({target})."
        )

    # SCRUM-495 / HG02: so từng loại (normalize). Mặc định chỉ cảnh báo —
    # align_flag_and_regen xử lý regen + flag; enforce_distribution=True mới trả error cứng.
    if enforce_distribution:
        expected = _expected_type_counts(plan)
        if expected:
            actual = Counter(
                _safe_normalize_question_type(q.question_type) for q in questions
            )
            if actual != expected:
                msg = (
                    "Phân bổ questionType không khớp approved plan: "
                    f"expected={dict(expected)} actual={dict(actual)}"
                )
                logger.warning("SCRUM-495 %s", msg)
                return msg

    return None
