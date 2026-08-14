"""Sinh interview plan từ JD + RAG context — JSON only output."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI

from config.settings import Settings
from helpers.experience_level_infer import require_experience_level_from_llm
from models.internal_schemas import (
    DifficultyDistributionItem,
    GeneratePlanRequest,
    GeneratePlanResponse,
    PlanCitationItem,
    QuestionGenerationPlan,
    QuestionTypeDistributionItem,
    RecommendedQuestionOutlineItem,
    SkillCoverageItem,
)
from services.json_output_parser import (
    build_json_fix_prompt,
    build_plan_validation_fix_prompt,
    extract_json_object,
    is_retryable_plan_error,
)
from services.rag_context_helpers import (
    JD_SOURCE_FILE,
    build_chunk_lookup,
    enrich_citations,
    ensure_jd_primary_citations,
    ensure_jd_primary_source_files,
    format_retrieved_context,
    parse_citations,
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

_PLAN_SYSTEM_PROMPT_BASE = """Bạn là chuyên gia lập kế hoạch phỏng vấn (interview planning expert).

## Mục tiêu
Tạo KẾ HOẠCH câu hỏi phỏng vấn (interview question plan), KHÔNG tạo câu hỏi cụ thể.
JD là nguồn CHÍNH; Knowledge ([HỆ THỐNG]/[HR] chunks) chỉ là nguồn PHỤ (policy, chuẩn nội bộ, tech doc).

## Nguồn suy luận (BẮT BUỘC)
Mọi field trong plan PHẢI do bạn suy luận từ:
1) jobDescription + hrNote + skills/loai_cau HR gửi (CHÍNH)
2) Context đã retrieve [HỆ THỐNG] / [HR] (PHỤ — nếu có)
KHÔNG được bỏ trống field, KHÔNG dùng giá trị mặc định ẩn, KHÔNG copy máy móc input HR nếu JD/RAG cho tín hiệu khác.
Không để Knowledge thay thế yêu cầu trong JD nếu mâu thuẫn.

## Quy tắc
1. Chỉ trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON.
2. Không bịa skills không liên quan đến JD hoặc skills được yêu cầu.
3. Mỗi coverage item phải gắn với skills yêu cầu hoặc trách nhiệm trong JD.
4. Citations: BẮT BUỘC có citation JD đầu tiên (knowledge_base=\"hr\", source_file=\"{jd_source}\", chunk_index=0). SCRUM-394: excerpt JD phải trích NGUYÊN VĂN từ jobDescription (substring liên quan skills/role) — giống excerpt Knowledge từ chunk; không paraphrase, không rỗng. Citation Knowledge chỉ thêm sau khi dùng chunk.
5. Mỗi coverage.source_files phải bắt đầu bằng \"{jd_source}\"; thêm file KB phía sau nếu dùng.
6. total_questions phải khớp số câu yêu cầu (so_cau); phân bổ question_type_distribution và difficulty_distribution phải cộng đúng total_questions.
7. {language_rule}
8. [HỆ THỐNG] và [HR] là context bổ sung và có thể trống; nếu thiếu context thì vẫn tạo plan dựa trên jobDescription, skills và hrNote.

## Field bắt buộc để map API (phải khớp yêu cầu HR)
- role_title: tên VỊ TRÍ tuyển dụng suy ra từ JD (ví dụ "Backend Developer") — KHÔNG phải danh sách skills.
- level: cùng giá trị difficulty (easy|medium|hard) — phải khớp difficulty HR yêu cầu.
- difficulty: cùng giá trị level.
- experience_level: BẮT BUỘC — cấp ứng viên mà bộ câu hỏi phù hợp (intern|junior|mid|senior|lead). KHÔNG được null hoặc bỏ trống.
  Suy ra từ JD/hrNote theo bảng:
  | Tín hiệu trong JD | experience_level |
  | Thực tập, intern, fresher | intern |
  | 0-1 năm, junior, mới ra trường | junior |
  | 2-4 năm, 1-3 năm, 3-5 năm kinh nghiệm | mid |
  | Senior, 5+ năm | senior |
  | Lead, Principal, Architect | lead |
  Ví dụ: "Backend Developer có 2–4 năm kinh nghiệm" → experience_level = "mid".
- total_questions: đúng số câu HR yêu cầu (so_cau).
- skills: ưu tiên skills HR gửi, bổ sung từ JD nếu thiếu và liên quan.
- question_type_distribution: phân bổ theo loại câu HR yêu cầu (loai_cau); chỉ dùng các type đã cho.

9. experience_level và level là field BẮT BUỘC — thiếu một trong hai thì JSON không hợp lệ.

## Schema JSON bắt buộc
{{
  "plan": {{
    "role_title": "string",
    "summary": "string",
    "difficulty": "easy|medium|hard",
    "level": "easy|medium|hard",
    "experience_level": "intern|junior|mid|senior|lead",
    "total_questions": 10,
    "skills": ["skill1"],
    "question_type_distribution": [
      {{"type": "technical", "count": 6, "reason": "string"}}
    ],
    "difficulty_distribution": [
      {{"difficulty": "medium", "count": 10}}
    ],
    "coverage": [
      {{"skill": "ASP.NET Core", "question_count": 3, "focus_areas": ["Web API"], "source_files": ["{jd_source}", "ten_file.pdf"]}}
    ],
    "recommended_question_outline": [
      {{
        "order": 1,
        "type": "technical",
        "difficulty": "medium",
        "skill": "ASP.NET Core",
        "focus_area": "Middleware",
        "goal": "string"
      }}
    ],
    "notes": "string",
    "citations": [
      {{
        "knowledge_base": "hr",
        "source_file": "{jd_source}",
        "chunk_index": 0,
        "excerpt": "doan trich ngan tu JD"
      }}
    ]
  }}
}}""".replace("{jd_source}", JD_SOURCE_FILE)


def _plan_system_prompt(language: str | None) -> str:
    return _PLAN_SYSTEM_PROMPT_BASE.format(
        language_rule=language_instruction_block(language)
    )


# Tương thích import cũ
PLAN_SYSTEM_PROMPT = _plan_system_prompt("Vietnamese")


class PlanGenerationService:
    def __init__(
        self,
        retrieval: RagRetrievalService,
        client: OpenAI,
        settings: Settings,
    ):
        self._retrieval = retrieval
        self._client = client
        self._settings = settings

    def _fail(
        self,
        start: float,
        error: str,
        stage: str,
        detail: str | None = None,
    ) -> GeneratePlanResponse:
        msg = detail or error
        return GeneratePlanResponse(
            success=False,
            error=error,
            detail=msg,
            stage=stage,
            exception_type="PlanGenerationError",
            errors=[msg],
            processing_time_ms=(time.time() - start) * 1000,
        )

    def generate(self, request: GeneratePlanRequest) -> GeneratePlanResponse:
        start = time.time()

        if not request.owner_id or not request.owner_id.strip():
            return self._fail(start, "ownerId là bắt buộc", "VALIDATION")

        if not request.job_description or not request.job_description.strip():
            return self._fail(start, "jobDescription là bắt buộc", "VALIDATION")

        try:
            # SCRUM-388: filter Selected docs + query JD+hrNote (instruction/focus)
            doc_ids = [str(d).strip() for d in (request.document_ids or []) if str(d).strip()]
            system_chunks, hr_chunks = self._retrieval.retrieve_for_job(
                request.job_description,
                request.owner_id.strip(),
                document_ids=doc_ids or None,
                query_extra=request.hr_note,
            )
        except Exception as exc:
            logger.exception("Retrieval failed")
            return self._fail(start, "Lỗi retrieval", "RETRIEVAL", str(exc))

        all_chunks = system_chunks + hr_chunks
        if not all_chunks:
            logger.warning(
                "No retrieved chunks for plan generation; falling back to JD only "
                "(owner_id=%s)",
                request.owner_id,
            )

        user_message = self._build_user_message(request, system_chunks, hr_chunks)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _plan_system_prompt(request.language)},
            {"role": "user", "content": user_message},
        ]

        try:
            raw = self._call_llm(messages)
        except Exception as exc:
            logger.exception("LLM call failed")
            return self._fail(start, "Lỗi sinh plan", "PLAN_GENERATION", f"Lỗi LLM: {exc}")

        chunk_lookup = build_chunk_lookup(all_chunks)
        plan, parse_error = self._parse_plan(raw, chunk_lookup, request)

        if parse_error and is_retryable_plan_error(parse_error):
            logger.warning("Plan generation failed, retry once: %s", parse_error)
            fix_prompt = (
                build_json_fix_prompt(raw)
                if parse_error.startswith(("LLM không", "Không parse", "JSON thiếu", "Không map"))
                else build_plan_validation_fix_prompt(parse_error, raw)
            )
            retry_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": fix_prompt},
            ]
            try:
                raw = self._call_llm(retry_messages)
            except Exception as exc:
                logger.exception("LLM retry call failed")
                return self._fail(
                    start,
                    "Lỗi sinh plan",
                    "PLAN_GENERATION",
                    f"Lỗi LLM khi retry: {exc}",
                )
            plan, parse_error = self._parse_plan(raw, chunk_lookup, request)

        if parse_error:
            return self._fail(start, "Lỗi sinh plan", "PLAN_GENERATION", parse_error)

        return GeneratePlanResponse(
            success=True,
            plan=plan,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._settings.chat_model,
            messages=messages,
            temperature=self._settings.temperature,
            stream=False,
        )
        return response.choices[0].message.content or ""

    def _build_user_message(
        self,
        request: GeneratePlanRequest,
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
        if request.document_ids:
            lines.append(
                "documentIds (Selected HR docs — ưu tiên coverage/source_files từ các doc này): "
                + ", ".join(str(d) for d in request.document_ids[:20])
            )
        if request.hr_note and request.hr_note.strip():
            lines.append(f"hrNote: {request.hr_note.strip()}")
        content_mode, code_templates = _parse_content_preferences(request.hr_note)
        lines.append(f"contentMode: {content_mode}")
        lines.append(f"enabledCodeTemplates: {', '.join(code_templates)}")

        lines.append(
            "\nHướng dẫn (SCRUM-392/394): JD là nguồn CHÍNH — citations BẮT BUỘC có "
            f'source_file="{JD_SOURCE_FILE}" đứng đầu; excerpt JD = đoạn NGUYÊN VĂN '
            "copy từ jobDescription (substring liên quan role/skills) — giống excerpt Knowledge. "
            "Knowledge chỉ PHỤ. Mỗi coverage.source_files bắt đầu bằng "
            f'"{JD_SOURCE_FILE}", thêm file KB sau nếu dùng. '
            "Citation chunk phải khớp source_file và chunk_index trong context. "
            "Nếu [HỆ THỐNG] hoặc [HR] không có chunk thì "
            "bỏ qua context đó, không coi đó là lỗi; nếu không có context nào thì "
            "vẫn giữ citation JD với excerpt từ jobDescription."
        )

        # Studio Plan UI / Apply settings
        hr = (request.hr_note or "").upper()
        natural = free_text_language_label(request.language)
        exclusive = "EXCLUSIVE_FOCUS=1" in hr or "EXCLUSIVE_FOCUS = 1" in hr
        strict = "STRICT_CORPUS=1" in hr or "STRICT_CORPUS = 1" in hr
        if "STUDIO_UI_PLAN" in hr or "APPLY_STUDIO_SETTINGS" in hr:
            coverage_rule = (
                "- coverage: 1–N items CHỈ ALLOWED_TOPICS / FOCUS_HINTS trong hrNote; "
                "KHÔNG thêm skill JD lan man (.NET/DB/FE nếu không thuộc topic); "
                "mỗi coverage có question_count + source_files từ context.\n"
                if exclusive
                else (
                    "- coverage: 1–6 skill CHỈ từ JD text + retrieved Selected chunks "
                    "(STRICT_CORPUS — không ép ≥4 skill JD); mỗi coverage có question_count + source_files.\n"
                    if strict
                    else (
                        "- coverage: ít nhất 4 skill từ JD+RAG, có question_count; "
                        "mỗi coverage PHẢI có source_files = danh sách source_file thật từ context [HỆ THỐNG]/[HR] "
                        "(nếu không có chunk thì source_files=[]).\n"
                    )
                )
            )
            lines.append(
                "\n[STUDIO_UI_PLAN / APPLY_STUDIO_SETTINGS — BẮT BUỘC cho UI Studio]\n"
                f"- difficulty_distribution: phân bổ easy/medium/hard cộng đúng {request.number_of_questions}, "
                "không dồn hết một mức (trừ khi TARGET_DIFFICULTY / instruction yêu cầu một mức).\n"
                "- question_type_distribution: ĐÚNG các type trong loai_cau "
                f"({types_text}), mỗi type count>0 nếu có trong loai_cau, reason bằng {natural}; "
                "tổng count = total_questions.\n"
                f"- recommended_question_outline: đúng {request.number_of_questions} item; "
                f"mỗi item có type/difficulty/skill/focus_area/goal (goal bằng {natural}).\n"
                + coverage_rule
                + "- citations: đủ các source_file đã dùng trong coverage.\n"
                f"- summary: 2–3 câu bằng {natural} mô tả cấu trúc buổi phỏng vấn"
                + (" — nêu rõ exclusive topic nếu EXCLUSIVE_FOCUS.\n" if exclusive else
                   " (Technical Foundations / System Design / Problem Solving / Behavioral).\n")
                + "- Nếu MODE=APPLY_STUDIO_SETTINGS: KHÔNG giữ total_questions/difficulty/types cũ nếu TARGET khác."
            )
            if exclusive or strict:
                lines.append(
                    "\n[STRICT/EXCLUSIVE — instruction THẮNG JD]: "
                    "Không giữ coverage cũ ngoài ALLOWED_TOPICS; role_title có thể giữ từ JD nhưng "
                    "summary/skills/coverage phải khớp instruction + corpus."
                )

        lines.append(
            f"\nMap API (BẮT BUỘC — bạn PHẢI suy luận, không bỏ trống): "
            f"role_title từ JD+RAG; level = difficulty = {request.difficulty}; "
            f"experience_level = intern|junior|mid|senior|lead từ JD/hrNote+RAG; "
            f"total_questions = {request.number_of_questions}; "
            f"skills = suy từ JD/hrNote/skills HR (ưu tiên HR, bổ sung từ JD nếu thiếu); "
            f"question_type_distribution phân bổ đúng loai_cau: {types_text}; "
            f"summary và notes bằng {natural} giải thích lý do chọn cấp độ và phân bổ."
        )
        lines.append("\n" + format_retrieved_context(system_chunks, hr_chunks))
        lines.append(
            f"\nTạo plan cho đúng {request.number_of_questions} câu hỏi. "
            "Trả về JSON theo schema (không tạo câu hỏi cụ thể)."
        )
        return "\n".join(lines)

    def _parse_plan(
        self,
        raw: str,
        chunk_lookup: dict[tuple[str, str, int], RetrievedChunk],
        request: GeneratePlanRequest,
    ) -> tuple[QuestionGenerationPlan | None, str | None]:
        data, parse_err = extract_json_object(raw)
        if parse_err or data is None:
            return None, parse_err or "LLM không trả về JSON hợp lệ."

        plan_data: Any = None
        if isinstance(data, dict):
            if "plan" in data:
                plan_data = data["plan"]
            elif "role_title" in data or "roleTitle" in data:
                plan_data = data
        if not isinstance(plan_data, dict):
            return None, "JSON thiếu object 'plan'."

        try:
            plan = _dict_to_plan(plan_data, chunk_lookup, request)
        except Exception as exc:
            return None, f"Không map được plan: {exc}"

        validation_error = _validate_plan(plan, request)
        if validation_error:
            return None, validation_error

        return plan, None


def _require_plan_str(plan_data: dict[str, Any], *keys: str, field_name: str) -> str:
    for key in keys:
        value = plan_data.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(
        f"Plan thiếu {field_name} — LLM phải suy luận từ JD, hrNote và RAG context."
    )


def _dict_to_plan(
    plan_data: dict[str, Any],
    chunk_lookup: dict[tuple[str, str, int], RetrievedChunk],
    request: GeneratePlanRequest,
) -> QuestionGenerationPlan:
    """Map LLM dict (snake_case hoặc camelCase) sang QuestionGenerationPlan."""
    role_title = _require_plan_str(plan_data, "role_title", "roleTitle", field_name="role_title")
    summary = _require_plan_str(plan_data, "summary", field_name="summary")

    total_raw = plan_data.get("total_questions") or plan_data.get("totalQuestions")
    if total_raw is None:
        raise ValueError(
            "Plan thiếu total_questions — LLM phải suy luận từ yêu cầu so_cau."
        )
    total_questions = int(total_raw)

    difficulty = _require_plan_str(plan_data, "difficulty", field_name="difficulty")
    level = _require_plan_str(plan_data, "level", field_name="level")

    skills_raw = plan_data.get("skills")
    if not isinstance(skills_raw, list) or not skills_raw:
        raise ValueError(
            "Plan thiếu skills — LLM phải suy luận từ JD, hrNote và RAG context."
        )
    skills = [str(s).strip() for s in skills_raw if str(s).strip()]
    if not skills:
        raise ValueError("Plan skills rỗng — LLM phải liệt kê ít nhất một skill.")

    type_dist_raw = plan_data.get("question_type_distribution") or plan_data.get(
        "questionTypeDistribution", []
    )
    type_dist = [
        QuestionTypeDistributionItem(
            type=str(item.get("type", "technical")),
            count=int(item.get("count", 0)),
            reason=str(item.get("reason", "")),
        )
        for item in type_dist_raw
        if isinstance(item, dict)
    ]

    diff_dist_raw = plan_data.get("difficulty_distribution") or plan_data.get(
        "difficultyDistribution", []
    )
    diff_dist = [
        DifficultyDistributionItem(
            difficulty=str(item.get("difficulty", "medium")),
            count=int(item.get("count", 0)),
        )
        for item in diff_dist_raw
        if isinstance(item, dict)
    ]

    coverage_raw = plan_data.get("coverage", [])
    coverage = []
    for item in coverage_raw:
        if not isinstance(item, dict):
            continue
        raw_sources = item.get("source_files", item.get("sourceFiles", [])) or []
        source_files: list[str] = []
        if isinstance(raw_sources, list):
            for s in raw_sources:
                name = str(s).strip()
                if name and name not in source_files:
                    source_files.append(name)
        coverage.append(
            SkillCoverageItem(
                skill=str(item.get("skill", "")),
                questionCount=int(
                    item.get("question_count", item.get("questionCount", 0))
                ),
                focusAreas=item.get("focus_areas", item.get("focusAreas", [])) or [],
                sourceFiles=source_files,
            )
        )

    outline_raw = plan_data.get("recommended_question_outline") or plan_data.get(
        "recommendedQuestionOutline", []
    )
    outline = [
        RecommendedQuestionOutlineItem(
            order=int(item.get("order", 0)),
            type=str(item.get("type", "technical")),
            difficulty=str(item.get("difficulty", "medium")),
            skill=str(item.get("skill", "")),
            focusArea=str(item.get("focus_area", item.get("focusArea", ""))),
            goal=str(item.get("goal", "")),
        )
        for item in outline_raw
        if isinstance(item, dict)
    ]

    raw_citations = plan_data.get("citations", [])
    citations = enrich_citations(parse_citations(raw_citations), chunk_lookup)
    # SCRUM-392/394: JD luôn đứng đầu + excerpt liên quan role/skills
    skills_hint = " ".join(
        str(s) for s in (plan_data.get("skills") or []) if s
    )
    plan_hint = " ".join(
        part
        for part in [
            str(plan_data.get("role_title") or plan_data.get("roleTitle") or ""),
            str(plan_data.get("summary") or ""),
            skills_hint,
        ]
        if part
    )
    citations = ensure_jd_primary_citations(
        citations, request.job_description, hint=plan_hint
    )

    # SCRUM-369: nếu coverage thiếu source_files → gán từ citations (round-robin / skill match)
    citation_files = [
        c.source_file.strip()
        for c in citations
        if getattr(c, "source_file", None) and str(c.source_file).strip()
    ]
    citation_files = list(dict.fromkeys(citation_files))
    if citation_files:
        enriched_coverage: list[SkillCoverageItem] = []
        for idx, cov in enumerate(coverage):
            files = list(cov.source_files or [])
            if not files:
                # ưu tiên citation có excerpt/skill gần skill name
                skill_l = (cov.skill or "").lower()
                matched = [
                    c.source_file.strip()
                    for c in citations
                    if c.source_file
                    and (
                        skill_l in (c.excerpt or "").lower()
                        or skill_l in c.source_file.lower()
                    )
                ]
                matched = list(dict.fromkeys(matched))
                files = matched[:2] if matched else [citation_files[idx % len(citation_files)]]
            # SCRUM-392: luôn lead bằng job-description
            files = ensure_jd_primary_source_files(files)
            enriched_coverage.append(
                cov.model_copy(update={"source_files": files})
            )
        coverage = enriched_coverage
    else:
        coverage = [
            cov.model_copy(
                update={"source_files": ensure_jd_primary_source_files(cov.source_files)}
            )
            for cov in coverage
        ]

    experience_raw = plan_data.get("experience_level") or plan_data.get("experienceLevel")
    experience_level = require_experience_level_from_llm(experience_raw)

    return QuestionGenerationPlan(
        roleTitle=role_title,
        summary=summary,
        difficulty=difficulty,
        level=level,
        experienceLevel=experience_level,
        totalQuestions=total_questions,
        skills=skills,
        questionTypeDistribution=type_dist,
        difficultyDistribution=diff_dist,
        coverage=coverage,
        recommendedQuestionOutline=outline,
        notes=str(plan_data.get("notes", "")),
        citations=citations,
    )


def _validate_plan(plan: QuestionGenerationPlan, request: GeneratePlanRequest) -> str | None:
    if plan.total_questions != request.number_of_questions:
        return (
            f"totalQuestions ({plan.total_questions}) "
            f"không khớp yêu cầu ({request.number_of_questions})."
        )

    if plan.question_type_distribution:
        type_sum = sum(item.count for item in plan.question_type_distribution)
        if type_sum != plan.total_questions:
            return (
                f"Tổng questionTypeDistribution ({type_sum}) "
                f"không khớp totalQuestions ({plan.total_questions})."
            )

    if plan.difficulty_distribution:
        diff_sum = sum(item.count for item in plan.difficulty_distribution)
        if diff_sum != plan.total_questions:
            return (
                f"Tổng difficultyDistribution ({diff_sum}) "
                f"không khớp totalQuestions ({plan.total_questions})."
            )

    if not plan.role_title.strip():
        return "Plan thiếu roleTitle."

    if not plan.summary.strip():
        return "Plan thiếu summary."

    if not plan.level.strip():
        return "Plan thiếu level."

    if plan.level != request.difficulty:
        return (
            f"level ({plan.level}) không khớp difficulty yêu cầu ({request.difficulty})."
        )

    if not plan.experience_level:
        return "Plan thiếu experienceLevel."

    if not plan.skills:
        return "Plan thiếu skills."

    if plan.difficulty != request.difficulty:
        return (
            f"difficulty/level ({plan.difficulty}) "
            f"không khớp yêu cầu ({request.difficulty})."
        )

    return None
