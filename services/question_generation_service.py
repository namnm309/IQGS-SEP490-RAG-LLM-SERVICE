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
    parse_citations,
    JD_SOURCE_FILE,
)
from services.rag_retrieval_service import RagRetrievalService
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
- Mỗi phần tử trong questions[] gắn đúng MỘT code_template_type và tuân thủ rule của template đó.
- Không tự đổi sang template khác cho cùng một câu; không Multiple Choice.
- Nội dung phù hợp Programming Language (từ JD/skills), Difficulty, Skills, JD, RAG context; ví dụ sát thực tế.
- Không tiết lộ đáp án trong question / code_snippet.
- Template cần code → phải có code_snippet thật (không placeholder: "viết tại đây", "TODO only", "complete here", stub 1 dòng comment).
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

## Mục tiêu
Tạo bộ câu hỏi phỏng vấn. Job Description (JD) là nguồn CHÍNH; tài liệu Knowledge ([HỆ THỐNG]/[HR] chunks) chỉ là nguồn PHỤ (policy, chuẩn nội bộ, tech doc).

## Quy tắc
1. Bám JD trước. Không bịa yêu cầu không có trong JD. Knowledge chỉ bổ sung; không thay JD nếu mâu thuẫn.
2. Mỗi câu hỏi phải bám difficulty và skills được yêu cầu.
3. Cân bằng loại câu hỏi theo question_types được yêu cầu.
4. {language_rule}
5. Chỉ trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON.
6. Mỗi câu hỏi BẮT BUỘC có citation JD đầu tiên: knowledge_base=\"hr\", source_file=\"{jd_source}\", chunk_index=0.
7. SCRUM-394: excerpt JD BẮT BUỘC là trích NGUYÊN VĂN ngắn từ jobDescription (substring thật), đúng đoạn skill/yêu cầu mà câu hỏi dựa vào — GIỐNG cách citation Knowledge trích từ chunk. KHÔNG paraphrase, KHÔNG để excerpt rỗng, KHÔNG lấy đại phần đầu JD nếu không liên quan.
8. Citation Knowledge (system/hr file) chỉ thêm SAU JD khi thật sự dùng đoạn context đó; excerpt phải là trích nguyên văn ngắn từ chunk.
9. sample_answer ưu tiên dựa trên JD + excerpt/citations. Nếu có code: giải thích văn xuôi rồi ```lang fence``` (trong string JSON); không gộp text+code một khối; không dùng nhãn "code:".
10. SCRUM-400: mỗi câu BẮT BUỘC có answer_method = "Text" | "Code" (Code = ứng viên nhập code; Text = trả lời văn xuôi).

## Schema JSON bắt buộc
{{
  "questions": [
    {{
      "question": "string",
      "question_type": "technical|behavioral|situational|system-design|problem-solving",
      "difficulty": "easy|medium|hard",
      "rationale": "string",
      "sample_answer": "string — giải thích văn xuôi rồi (nếu có) ```lang\\ncode\\n```; không gộp text vào fence",
      "image_hint": "string — gợi ý HR nên tìm/đính kèm hình hoặc diagram nào (KHÔNG gen ảnh)",
      "answer_method": "Text|Code",
      "citations": [
        {{
          "knowledge_base": "hr",
          "source_file": "{jd_source}",
          "chunk_index": 0,
          "excerpt": "doan trich ngan tu JD"
        }}
      ]
    }}
  ]
}}""".replace("{jd_source}", JD_SOURCE_FILE)

_QUESTIONS_FROM_PLAN_SYSTEM_PROMPT_BASE = """Bạn là chuyên gia sinh câu hỏi phỏng vấn (interview question generator).

## Mục tiêu
Sinh câu hỏi phỏng vấn từ APPROVED PLAN đã được HR duyệt. KHÔNG thay đổi plan.
JD là nguồn CHÍNH; Knowledge ([HỆ THỐNG]/[HR] chunks) là nguồn PHỤ.

## Quy tắc
1. Tuân thủ chính xác approvedPlan: totalQuestions, questionTypeDistribution, coverage, recommendedQuestionOutline.
2. Không sửa đổi hoặc tái lập kế hoạch — chỉ sinh câu hỏi.
3. Chỉ trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON.
4. Trong string JSON: escape đúng chuẩn (\\\\ \\\\\\\" \\\\n). Không dùng \\s, \\a, hay backslash đơn trước chữ/số.
5. {language_rule}
6. [HỆ THỐNG] và [HR] là context bổ sung và có thể trống; nếu thiếu context thì vẫn sinh câu hỏi từ approvedPlan, jobDescription và hrNote.
7. Mỗi câu BẮT BUỘC có citation JD đầu tiên: knowledge_base=\"hr\", source_file=\"{jd_source}\", chunk_index=0.
8. SCRUM-394: excerpt JD BẮT BUỘC trích NGUYÊN VĂN từ jobDescription (substring), đúng đoạn liên quan skill/focusArea của câu — giống citation Knowledge. Không paraphrase, không rỗng.
9. Citation Knowledge chỉ thêm sau JD khi dùng chunk retrieve; nếu không có chunk context thì chỉ citation JD.
10. sample_answer ưu tiên dựa trên JD / approvedPlan; bổ sung từ Knowledge khi có.
    Nếu có code đáp án: (1) 1–3 câu giải thích ngoài fence (2) rồi ```csharp|sql|python|... code ``` trong giá trị string.
    Cấm gộp giải thích + code thành một khối; cấm nhãn "code:"; cấm copy code_snippet đề bài vào sample_answer.
11. Thứ tự câu hỏi theo recommendedQuestionOutline nếu có.
12. SCRUM-396: mỗi câu có code_template_type phải tuân thủ rule template (xem user prompt). Không đổi template của câu.
13. code_snippet BẮT BUỘC (non-empty, code thật) khi code_template_type là CODE_COMPLETION|BUG_DETECTION|REFACTORING|TEST_CASE_DESIGN|PERFORMANCE_ANALYSIS.
14. SYSTEM_DESIGN: code_snippet optional; ưu tiên image_hint kiến trúc. Cấm placeholder ("viết tại đây", "TODO only", stub 1 dòng).
15. Escape JSON: dùng \\\\n để sau parse thành newline thật — không để chuỗi chữ \\\\n hiển thị trên UI.
16. SCRUM-400: mỗi câu BẮT BUỘC có answer_method = "Text" | "Code".
    - Code: ứng viên phải viết/sửa/phân tích code (CODE_COMPLETION, BUG_DETECTION, REFACTORING, TEST_CASE_DESIGN, PERFORMANCE_ANALYSIS).
    - Text: lý thuyết, behavioral, situational, SYSTEM_DESIGN giải thích kiến trúc (không bắt nhập code).

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
      "sample_answer": "string — giải thích văn xuôi rồi (nếu có) ```lang\\ncode\\n```; không gộp text vào fence",
      "evaluation_criteria": ["string"],
      "code_template_type": "CODE_COMPLETION|BUG_DETECTION|REFACTORING|TEST_CASE_DESIGN|PERFORMANCE_ANALYSIS|SYSTEM_DESIGN",
      "code_snippet": "string — bắt buộc với template code (không SYSTEM_DESIGN); skeleton/code thật multi-line",
      "image_hint": "string — gợi ý HR nên tìm/đính kèm hình hoặc diagram nào (KHÔNG gen ảnh)",
      "answer_method": "Text|Code",
      "citations": [
        {{
          "knowledge_base": "hr",
          "source_file": "{jd_source}",
          "chunk_index": 0,
          "excerpt": "doan trich ngan tu JD"
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


# Sinh batch để tránh timeout + JSON quá dài khi totalQuestions lớn (vd 30).
_QUESTION_BATCH_SIZE = 10
_BATCH_THRESHOLD = 12


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
            system_chunks, hr_chunks = self._retrieval.retrieve_for_job(
                request.job_description,
                request.owner_id.strip(),
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
            system_chunks, hr_chunks = self._retrieval.retrieve_for_job(
                request.job_description,
                request.owner_id.strip(),
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
            "\nHướng dẫn (SCRUM-392/394): JD là nguồn CHÍNH — mỗi câu BẮT BUỘC có citation "
            f'source_file="{JD_SOURCE_FILE}" (knowledge_base=hr, chunk_index=0) đứng đầu. '
            "excerpt JD phải là đoạn NGUYÊN VĂN copy từ jobDescription ở trên (substring), "
            "chỉ rõ phần JD mà câu hỏi dựa vào — giống excerpt của citation Knowledge. "
            "Knowledge chunks chỉ là PHỤ — citation file KB thêm sau JD khi thật sự dùng đoạn đó. "
            "Citation chunk phải khớp source_file/chunk_index trong context bên dưới."
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
        mode, templates = _parse_content_preferences(request.hr_note)
        lines.append(f"contentMode: {mode}")
        lines.append(f"enabledCodeTemplates: {', '.join(templates)}")
        lines.append("Nếu mode=TheoryOnly: chỉ câu hỏi lý thuyết, không tạo code_snippet.")
        lines.append(
            "Nếu mode=CodeOnly: mỗi câu cần code_template_type hợp lệ trong enabledCodeTemplates; "
            "template code bắt buộc có code_snippet thật (không placeholder)."
        )
        lines.append("Nếu mode=Mixed: trộn câu lý thuyết và câu code, cân bằng theo enabledCodeTemplates.")
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
            "Hướng dẫn (SCRUM-392/394): JD là nguồn CHÍNH — mỗi câu BẮT BUỘC citation "
            f'source_file="{JD_SOURCE_FILE}" đứng đầu. '
            "excerpt JD = trích nguyên văn từ jobDescription (substring liên quan skill/focus). "
            "Knowledge chỉ PHỤ — thêm citation chunk sau JD khi dùng context; "
            "nếu không có chunk thì chỉ citation JD. "
            "Citation chunk phải khớp source_file/chunk_index.",
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
            # SCRUM-392/394: JD đứng đầu + excerpt nguyên văn / liên quan câu hỏi
            rationale = str(item.get("rationale", ""))
            skill_val = str(item.get("skill") or "").strip()
            focus_val = str(
                item.get("focus_area") or item.get("focusArea") or ""
            ).strip()
            jd_hint = " ".join(
                part for part in [q_text, skill_val, focus_val, rationale] if part
            )
            citations = ensure_jd_primary_citations(
                citations, job_description, hint=jd_hint
            )

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
                if q.code_template_type not in enabled_templates:
                    q.code_template_type = enabled_templates[idx % len(enabled_templates)]

                # Unescape \\n + bỏ placeholder stub từ LLM
                q.code_snippet = _harden_code_snippet(q.code_snippet, q.code_template_type)

                needs_snippet = (q.code_template_type or "") in _TEMPLATES_REQUIRING_SNIPPET
                if needs_snippet and not q.code_snippet:
                    # CodeOnly luôn fallback; Mixed cũng fallback nếu template bắt buộc snippet
                    if content_mode == "CodeOnly" or needs_snippet:
                        q.code_snippet = _default_snippet_for_template(q.code_template_type or "") or None

                # SYSTEM_DESIGN: snippet optional — clear nếu chỉ là placeholder đã harden thành None
                if q.code_template_type == "SYSTEM_DESIGN" and not (q.code_snippet or "").strip():
                    q.code_snippet = None

                # SCRUM-400: harden answer_method theo mode + template
                if content_mode == "CodeOnly":
                    q.answer_method = "Code"
                elif not q.answer_method:
                    q.answer_method = _infer_answer_method(q.code_template_type, q.code_snippet)
                elif q.answer_method == "Text" and needs_snippet and (q.code_snippet or "").strip():
                    # Template code bắt buộc snippet → Candidate cần ô code
                    q.answer_method = "Code"

            # Mọi mode: luôn có image_hint gợi ý cho HR
            if not (q.image_hint or "").strip():
                q.image_hint = _default_image_hint(q.code_template_type, q.question_type)

            if not q.answer_method:
                q.answer_method = _infer_answer_method(q.code_template_type, q.code_snippet)
        return questions, None

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
                        q.evaluation_criteria = [str(x) for x in eval_raw]
                    tpl_val = item.get("code_template_type", item.get("codeTemplateType"))
                    if tpl_val:
                        q.code_template_type = str(tpl_val).strip()
                    snippet_val = item.get("code_snippet", item.get("codeSnippet"))
                    if snippet_val:
                        q.code_snippet = str(snippet_val)
                    hint_val = item.get("image_hint", item.get("imageHint"))
                    if hint_val:
                        q.image_hint = str(hint_val).strip()

        # Gán order từ outline nếu LLM thiếu
        if approved_plan.recommended_question_outline:
            outline_by_order = {
                o.order: o for o in approved_plan.recommended_question_outline
            }
            for idx, q in enumerate(questions):
                if q.order is None and idx < len(
                    approved_plan.recommended_question_outline
                ):
                    outline = approved_plan.recommended_question_outline[idx]
                    q.order = outline.order
                    if not q.skill:
                        q.skill = outline.skill
                    if not q.focus_area:
                        q.focus_area = outline.focus_area
                elif q.order is not None and q.order in outline_by_order:
                    outline = outline_by_order[q.order]
                    if not q.skill:
                        q.skill = outline.skill
                    if not q.focus_area:
                        q.focus_area = outline.focus_area

        validation_error = _validate_questions_against_plan(
            questions,
            approved_plan,
            expected_count=expected_count,
            strict_count=expected_count is None,
        )
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
