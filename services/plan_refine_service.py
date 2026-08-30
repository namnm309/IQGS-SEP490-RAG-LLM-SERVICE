"""SCRUM-420: Refine plan — LLM trả PlanPatch (delta), không full regen."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI

from config.settings import Settings
from models.internal_schemas import (
    PlanPatch,
    RefinePlanRequest,
    RefinePlanResponse,
    SkillCoverageItem,
)
from services.json_output_parser import build_json_fix_prompt, extract_json_object
from services.plan_provenance_validator import apply_provenance_to_plan_dict
from services.rag_context_helpers import format_retrieved_context
from services.rag_retrieval_service import RagRetrievalService
from vectorstores.base import RetrievedChunk
from helpers.language_prompt import free_text_language_label, language_instruction_block

logger = logging.getLogger(__name__)

_REFINE_SYSTEM = """Bạn là chuyên gia refine kế hoạch phỏng vấn (interview plan patch).

## Mục tiêu
Nhận BASELINE PLAN hiện tại + instruction HR + context retrieve.
Chỉ trả JSON **patch** (delta) — KHÔNG trả full plan.
Phần plan KHÔNG liên quan instruction → KHÔNG đưa vào patch (server giữ từ baseline).

## Nguồn suy luận (waterfall — BẮT BUỘC ghi nhận khi đổi coverage)
1) HR — JD + chunk [HR] Selected
2) SYSTEM — chunk [HỆ THỐNG] Admin KB (vd. git.pdf)
3) LLM — suy luận cuối; mọi mục LLM cần lý do trong patch notes

## Quy tắc patch
- replaceCoverage: CHỈ khi instruction đổi skill/focus/coverage; mỗi item có source_files từ context thật
- replaceSkills: khi đổi danh sách skill theo instruction
- appendOutline / replaceOutline: khi cần outline mới khớp total_questions
- replaceQuestionTypeDistribution / replaceDifficultyDistribution: khi đổi loại câu hoặc độ khó
- updateSummary: khi summary cần phản ánh thay đổi
- replaceCitations: citations mới từ context đã dùng
- instructionApplied: tóm tắt ngắn instruction đã áp dụng

## Schema JSON (chỉ patch — field null/omit = giữ baseline)
{{
  "patch": {{
    "replaceCoverage": [...],
    "replaceSkills": ["..."],
    "replaceOutline": [...],
    "replaceQuestionTypeDistribution": [...],
    "replaceDifficultyDistribution": [...],
    "updateSummary": "...",
    "replaceCitations": [...],
    "instructionApplied": "..."
  }}
}}

Chỉ trả JSON hợp lệ, không markdown."""


def _has_focus_hints(hr_note: str | None) -> bool:
    note = (hr_note or "").upper()
    return "FOCUS_HINTS:" in note or "ALLOWED_TOPICS:" in note


def _is_exclusive(hr_note: str | None) -> bool:
    note = (hr_note or "").upper()
    return "EXCLUSIVE_FOCUS=1" in note or "EXCLUSIVE_FOCUS = 1" in note


class PlanRefineService:
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
    ) -> RefinePlanResponse:
        msg = detail or error
        return RefinePlanResponse(
            success=False,
            error=error,
            detail=msg,
            stage=stage,
            exception_type="PlanRefineError",
            errors=[msg],
            processing_time_ms=(time.time() - start) * 1000,
        )

    def refine(self, request: RefinePlanRequest) -> RefinePlanResponse:
        start = time.time()

        if not request.owner_id or not request.owner_id.strip():
            return self._fail(start, "ownerId là bắt buộc", "VALIDATION")
        if not request.job_description or not request.job_description.strip():
            return self._fail(start, "jobDescription là bắt buộc", "VALIDATION")
        if request.baseline_plan is None:
            return self._fail(start, "baselinePlan là bắt buộc", "VALIDATION")

        top_k_system = self._settings.top_k_system
        if _has_focus_hints(request.hr_note):
            top_k_system = max(top_k_system, 8)

        try:
            doc_ids = [str(d).strip() for d in (request.document_ids or []) if str(d).strip()]
            system_chunks, hr_chunks = self._retrieval.retrieve_for_job(
                request.job_description,
                request.owner_id.strip(),
                document_ids=doc_ids or None,
                query_extra=request.hr_note,
                top_k_system=top_k_system,
            )
        except Exception as exc:
            logger.exception("Refine retrieval failed")
            return self._fail(start, "Lỗi retrieval", "RETRIEVAL", str(exc))

        all_chunks = system_chunks + hr_chunks
        user_message = self._build_user_message(request, system_chunks, hr_chunks)

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": _REFINE_SYSTEM
                + "\n"
                + language_instruction_block(request.language),
            },
            {"role": "user", "content": user_message},
        ]

        try:
            raw = self._call_llm(messages)
        except Exception as exc:
            logger.exception("Refine LLM failed")
            return self._fail(start, "Lỗi refine plan", "PLAN_REFINE", f"Lỗi LLM: {exc}")

        patch, parse_error = self._parse_patch(raw)
        if parse_error:
            logger.warning("Patch parse failed, retry once: %s", parse_error)
            retry_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": build_json_fix_prompt(raw)},
            ]
            try:
                raw = self._call_llm(retry_messages)
                patch, parse_error = self._parse_patch(raw)
            except Exception as exc:
                return self._fail(start, "Lỗi refine plan", "PLAN_REFINE", str(exc))

        if parse_error or patch is None:
            return self._fail(
                start,
                "Lỗi refine plan",
                "PLAN_REFINE",
                parse_error or "Không parse được patch",
            )

        # Preview merge + provenance trên patch coverage (optional enrich)
        if patch.replace_coverage:
            enriched_cov = []
            for item in patch.replace_coverage:
                cov_dict = item.model_dump(by_alias=True)
                enriched_cov.append(
                    SkillCoverageItem.model_validate(cov_dict)
                )
            patch = patch.model_copy(update={"replace_coverage": enriched_cov})

        return RefinePlanResponse(
            success=True,
            patch=patch,
            processing_time_ms=(time.time() - start) * 1000,
        )

    def preview_merged_with_provenance(
        self,
        baseline: dict[str, Any],
        patch: PlanPatch,
        *,
        chunks: list[RetrievedChunk],
        job_description: str,
    ) -> dict[str, Any]:
        """Merge baseline + patch rồi chạy provenance — dùng test/debug."""
        merged = _merge_baseline_patch(baseline, patch.model_dump(by_alias=True))
        return apply_provenance_to_plan_dict(
            merged, chunks=chunks, job_description=job_description
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
        request: RefinePlanRequest,
        system_chunks: list[RetrievedChunk],
        hr_chunks: list[RetrievedChunk],
    ) -> str:
        baseline = request.baseline_plan
        if hasattr(baseline, "model_dump"):
            baseline_json = json.dumps(
                baseline.model_dump(by_alias=True), ensure_ascii=False, indent=2
            )
        elif isinstance(baseline, dict):
            baseline_json = json.dumps(baseline, ensure_ascii=False, indent=2)
        else:
            baseline_json = json.dumps(baseline, ensure_ascii=False, default=str, indent=2)

        types_text = ", ".join(request.question_types)
        natural = free_text_language_label(request.language)
        exclusive = _is_exclusive(request.hr_note)

        lines = [
            "[BASELINE PLAN — GIỮ NGUYÊN PHẦN KHÔNG ĐỤNG INSTRUCTION]",
            baseline_json,
            "",
            "[YÊU CẦU REFINE]",
            f"so_cau: {request.number_of_questions}",
            f"difficulty: {request.difficulty}",
            f"loai_cau: {types_text}",
            language_instruction_block(request.language),
        ]
        if request.experience_level:
            lines.append(
                f"experienceLevel (BẮT BUỘC giữ): {request.experience_level}"
            )
        if request.hr_note and request.hr_note.strip():
            lines.append(f"hrNote: {request.hr_note.strip()}")

        lines.append(
            "\n[QUY TẮC REFINE — SCRUM-420]\n"
            "- Instruction THẮNG baseline khi xung đột.\n"
            "- Cho phép dùng chunk [HỆ THỐNG] Admin trong coverage khi retrieve có (vd. git syntax).\n"
            "- Waterfall: HR → SYSTEM → LLM; field suy luận LLM cần lý do trong instructionApplied.\n"
            f"- Chỉ trả patch JSON; total_questions patch phải khớp {request.number_of_questions} nếu đổi coverage/outline.\n"
            f"- summary/goal/reason bằng {natural}."
        )
        if exclusive:
            lines.append(
                "EXCLUSIVE: replaceCoverage CHỈ ALLOWED_TOPICS; drop skill ngoài list."
            )

        lines.append("\n" + format_retrieved_context(system_chunks, hr_chunks))
        lines.append(
            "\nTrả về JSON patch theo schema. Không trả full plan."
        )
        return "\n".join(lines)

    def _parse_patch(self, raw: str) -> tuple[PlanPatch | None, str | None]:
        data, parse_err = extract_json_object(raw)
        if parse_err or data is None:
            return None, parse_err or "LLM không trả JSON hợp lệ."

        patch_data: Any = None
        if isinstance(data, dict):
            if "patch" in data:
                patch_data = data["patch"]
            elif any(
                k in data
                for k in (
                    "replaceCoverage",
                    "replace_coverage",
                    "updateSummary",
                    "update_summary",
                    "instructionApplied",
                    "instruction_applied",
                )
            ):
                patch_data = data

        if not isinstance(patch_data, dict):
            return None, "JSON thiếu object 'patch'."

        try:
            normalized = _normalize_patch_dict(patch_data)
            patch = PlanPatch.model_validate(normalized)
        except Exception as exc:
            return None, f"Không map patch: {exc}"

        if not patch.has_changes():
            return None, "Patch rỗng — LLM phải trả ít nhất một field delta."

        return patch, None


def _normalize_patch_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Chuẩn snake → alias camel cho pydantic."""
    key_map = {
        "replace_coverage": "replaceCoverage",
        "replace_skills": "replaceSkills",
        "replace_outline": "replaceOutline",
        "replace_question_type_distribution": "replaceQuestionTypeDistribution",
        "replace_difficulty_distribution": "replaceDifficultyDistribution",
        "update_summary": "updateSummary",
        "replace_citations": "replaceCitations",
        "instruction_applied": "instructionApplied",
    }
    out: dict[str, Any] = {}
    for k, v in data.items():
        out[key_map.get(k, k)] = v
    return out


def _merge_baseline_patch(baseline: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Merge đơn giản — mirror logic BE PlanMergeService."""
    merged = dict(baseline)
    if patch.get("replaceCoverage"):
        merged["coverage"] = patch["replaceCoverage"]
    if patch.get("replaceSkills"):
        merged["skills"] = patch["replaceSkills"]
    if patch.get("replaceOutline"):
        merged["recommendedQuestionOutline"] = patch["replaceOutline"]
    if patch.get("replaceQuestionTypeDistribution"):
        merged["questionTypeDistribution"] = patch["replaceQuestionTypeDistribution"]
    if patch.get("replaceDifficultyDistribution"):
        merged["difficultyDistribution"] = patch["replaceDifficultyDistribution"]
    if patch.get("updateSummary"):
        merged["summary"] = patch["updateSummary"]
    if patch.get("replaceCitations"):
        merged["citations"] = patch["replaceCitations"]
    return merged
