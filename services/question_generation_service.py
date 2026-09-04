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
    GeneratedQuestionItem,
    GenerateQuestionsFromPlanRequest,
    GenerateQuestionsFromPlanResponse,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    QuestionGenerationPlan,
    RubricCriterionItem,
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
from helpers.language_prompt import free_text_language_label, language_instruction_block

logger = logging.getLogger(__name__)
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


def _build_plan_retrieve_query(
    plan: QuestionGenerationPlan, hr_note: str | None
) -> str:
    """SCRUM-421: embed skill/focus từ plan — không chỉ JD."""
    parts: list[str] = []
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

        questions = apply_provenance_to_questions(
            questions,
            chunks=all_chunks,
            job_description=request.job_description,
        )

        return GenerateQuestionsFromPlanResponse(
            success=parse_error is None and len(questions) > 0,
            questions=questions,
            processing_time_ms=(time.time() - start) * 1000,
            error=parse_error,
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

        # Re-number + validate total (nới lỏng)
        merged.sort(key=lambda q: q.order if q.order is not None else 0)
        for i, q in enumerate(merged, start=1):
            q.order = i

        validation_error = _validate_questions_against_plan(
            merged, request.approved_plan, strict_count=False
        )
        if validation_error and len(merged) < max(1, (total * 2) // 3):
            return merged, validation_error
        if validation_error:
            logger.warning("Plan validation soft-accept: %s", validation_error)
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
        plan_json = request.approved_plan.model_dump(by_alias=True, mode="json")
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
            # SCRUM-429: Studio regen — tránh trùng câu khác trong cùng plan
            if "STUDIO_REGEN=1" in request.hr_note or "AVOID_QUESTIONS=" in request.hr_note:
                lines.append(
                    "STUDIO_REGEN / AVOID_QUESTIONS: Câu mới PHẢI khác ý và cấu trúc các mục "
                    "trong AVOID_QUESTIONS (phân tách |#|). Không paraphrase gần như giống; "
                    "đổi góc hỏi / ví dụ / yêu cầu cụ thể trong khi vẫn bám skill/focus/goal của outline."
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
                for i, item in enumerate(items):
                    if i >= len(questions) or not isinstance(item, dict):
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

        # Gán order / skill / focus / answerMethod từ outline nếu LLM thiếu
        if approved_plan.recommended_question_outline:
            outline_by_order = {
                o.order: o for o in approved_plan.recommended_question_outline
            }
            for idx, q in enumerate(questions):
                outline = None
                if q.order is None and idx < len(
                    approved_plan.recommended_question_outline
                ):
                    outline = approved_plan.recommended_question_outline[idx]
                    q.order = outline.order
                elif q.order is not None and q.order in outline_by_order:
                    outline = outline_by_order[q.order]
                if outline is None:
                    continue
                if not q.skill:
                    q.skill = outline.skill
                if not q.focus_area:
                    q.focus_area = outline.focus_area
                # HR preview: ưu tiên answerMethod trên outline.
                # Text = cách trả lời — KHÔNG xóa code_snippet đề nếu template code-heavy.
                preferred = _normalize_answer_method(
                    getattr(outline, "answer_method", None)
                )
                _apply_outline_answer_method(q, preferred)

        # SCRUM-427: rationale khóa = outline.goal
        self._seed_rationale_from_outline(questions, approved_plan)

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
        )
        mode, _ = _parse_content_preferences(hr_note)
        for q in questions:
            _align_code_to_question(q, mode)
            _ensure_code_heavy_has_snippet(q)
        return questions, validation_error


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

    if expected_count is None and plan.question_type_distribution:
        expected = Counter(
            {getattr(item, "type"): item.count for item in plan.question_type_distribution}
        )
        actual = Counter(q.question_type for q in questions)
        if actual != expected:
            logger.warning(
                "questionType distribution lệch: expected=%s actual=%s",
                dict(expected),
                dict(actual),
            )
            if sum(actual.values()) != plan.total_questions and abs(
                sum(actual.values()) - plan.total_questions
            ) > 2:
                return "Phân bổ questionType không khớp approved plan."

    return None
