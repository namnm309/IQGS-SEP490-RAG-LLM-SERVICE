"""Sinh câu hỏi phỏng vấn từ JD hoặc approved plan — JSON only output."""
from __future__ import annotations

import json
import logging
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
    format_retrieved_context,
    parse_citations,
)
from services.rag_retrieval_service import RagRetrievalService
from vectorstores.base import RetrievedChunk

logger = logging.getLogger(__name__)

INTERVIEW_SYSTEM_PROMPT = """Bạn là chuyên gia thiết kế câu hỏi phỏng vấn kỹ thuật.

## Mục tiêu
Tạo bộ câu hỏi phỏng vấn dựa trên [HỆ THỐNG] (rubric, question bank, tài liệu kỹ thuật) và [HR] (JD, policy, form).

## Quy tắc
1. Chỉ dùng thông tin trong [HỆ THỐNG] và [HR]. Không bịa yêu cầu không có trong JD.
2. Mỗi câu hỏi phải bám difficulty và skills được yêu cầu.
3. Cân bằng loại câu hỏi theo question_types được yêu cầu.
4. Trả lời BẰNG TIẾNG VIỆT.
5. Chỉ trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON.
6. Mỗi câu hỏi phải có ít nhất 1 citation nếu có tài liệu liên quan trong context.
7. excerpt phải là trích nguyên văn ngắn từ đoạn context (không paraphrase).
8. sample_answer chỉ được suy từ excerpt/citations; nếu không đủ căn cứ: "Không đủ căn cứ trong tài liệu."

## Schema JSON bắt buộc
{
  "questions": [
    {
      "question": "string",
      "question_type": "technical|behavioral|situational|system-design|problem-solving",
      "difficulty": "easy|medium|hard",
      "rationale": "string",
      "sample_answer": "string",
      "citations": [
        {
          "knowledge_base": "system|hr",
          "source_file": "ten_file",
          "chunk_index": 0,
          "excerpt": "đoạn trích ngắn nguyên văn"
        }
      ]
    }
  ]
}"""

QUESTIONS_FROM_PLAN_SYSTEM_PROMPT = """Bạn là chuyên gia sinh câu hỏi phỏng vấn (interview question generator).

## Mục tiêu
Sinh câu hỏi phỏng vấn từ APPROVED PLAN đã được HR duyệt. KHÔNG thay đổi plan.

## Quy tắc
1. Tuân thủ chính xác approvedPlan: totalQuestions, questionTypeDistribution, coverage, recommendedQuestionOutline.
2. Không sửa đổi hoặc tái lập kế hoạch — chỉ sinh câu hỏi.
3. Chỉ trả về JSON hợp lệ, không markdown, không giải thích ngoài JSON.
4. Trả lời bằng tiếng Việt.
5. [HỆ THỐNG] và [HR] là context bổ sung và có thể trống; nếu thiếu context thì vẫn sinh câu hỏi từ approvedPlan, jobDescription và hrNote.
6. Citations tham chiếu chunks retrieve khi context hỗ trợ câu hỏi; nếu không có chunk context, trả citations là [].
7. sample_answer ưu tiên dựa trên context đã retrieve; nếu không có context thì dựa trên approvedPlan và jobDescription.
8. Thứ tự câu hỏi theo recommendedQuestionOutline nếu có.

## Schema JSON bắt buộc
{
  "questions": [
    {
      "order": 1,
      "question": "string",
      "question_type": "technical|behavioral|situational|system-design|problem-solving",
      "difficulty": "easy|medium|hard",
      "skill": "string",
      "focus_area": "string",
      "rationale": "string",
      "sample_answer": "string",
      "evaluation_criteria": ["string"],
      "citations": [
        {
          "knowledge_base": "system|hr",
          "source_file": "ten_file",
          "chunk_index": 0,
          "excerpt": "đoạn trích ngắn"
        }
      ]
    }
  ]
}"""


class QuestionGenerationService:
    def __init__(
        self,
        retrieval: RagRetrievalService,
        client: OpenAI,
        settings: Settings,
    ):
        self._retrieval = retrieval
        self._client = client
        self._chat_model = settings.chat_model
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
            {"role": "system", "content": INTERVIEW_SYSTEM_PROMPT},
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
        questions, parse_error = self._parse_questions(raw, chunk_lookup)

        if parse_error and is_retryable_json_error(parse_error):
            raw, parse_error, questions = self._retry_parse(
                messages, raw, chunk_lookup, self._parse_questions
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

        user_message = self._build_from_plan_user_message(
            request, system_chunks, hr_chunks
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": QUESTIONS_FROM_PLAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        raw, llm_error = self._call_llm_safe(messages)
        if llm_error:
            return GenerateQuestionsFromPlanResponse(
                success=False,
                error=llm_error,
                processing_time_ms=(time.time() - start) * 1000,
            )

        chunk_lookup = build_chunk_lookup(all_chunks)
        questions, parse_error = self._parse_questions_from_plan(
            raw, chunk_lookup, request.approved_plan
        )

        if parse_error and is_retryable_json_error(parse_error):
            raw, parse_error, questions = self._retry_parse(
                messages,
                raw,
                chunk_lookup,
                lambda r, lk: self._parse_questions_from_plan(
                    r, lk, request.approved_plan
                ),
            )

        return GenerateQuestionsFromPlanResponse(
            success=parse_error is None and len(questions) > 0,
            questions=questions,
            processing_time_ms=(time.time() - start) * 1000,
            error=parse_error,
        )

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content or ""

    def _call_llm_safe(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, str | None]:
        try:
            return self._call_llm(messages), None
        except Exception as exc:
            logger.exception("LLM call failed")
            return "", f"Lỗi LLM: {exc}"

    def _retry_parse(
        self,
        messages: list[dict[str, str]],
        raw: str,
        chunk_lookup: dict[tuple[str, str, int], RetrievedChunk],
        parse_fn: Any,
    ) -> tuple[str, str | None, list[GeneratedQuestionItem]]:
        logger.warning("LLM JSON parse failed, retry once")
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": build_json_fix_prompt(raw)},
        ]
        try:
            raw = self._call_llm(retry_messages)
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
        ]
        if skills_text:
            lines.append(f"skills: {skills_text}")
        if request.hr_note and request.hr_note.strip():
            lines.append(f"hrNote: {request.hr_note.strip()}")

        lines.append(
            "\nHướng dẫn: mỗi citation phải khớp đúng source_file và chunk_index "
            "trong từng đoạn dưới. excerpt trích nguyên văn từ đoạn đó."
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
    ) -> str:
        plan_json = request.approved_plan.model_dump(by_alias=True, mode="json")

        lines = [
            "[YÊU CẦU]",
            f"ownerId: {request.owner_id}",
            f"jobDescription: {request.job_description}",
        ]
        if request.hr_note and request.hr_note.strip():
            lines.append(f"hrNote: {request.hr_note.strip()}")
        lines.extend([
            "",
            "[APPROVED PLAN — KHÔNG ĐƯỢC THAY ĐỔI]",
            json.dumps(plan_json, ensure_ascii=False, indent=2),
            "",
            "Hướng dẫn: sinh đúng số câu theo plan. Nếu có context thì citations khớp "
            "source_file/chunk_index; nếu không có context nào thì trả citations là [].",
            "",
            format_retrieved_context(system_chunks, hr_chunks),
            "",
            f"Tạo đúng {request.approved_plan.total_questions} câu hỏi theo approved plan. "
            "Trả về JSON theo schema.",
        ])
        return "\n".join(lines)

    def _parse_questions(
        self,
        raw: str,
        chunk_lookup: dict[tuple[str, str, int], RetrievedChunk],
    ) -> tuple[list[GeneratedQuestionItem], str | None]:
        data, parse_err = extract_json_object(raw)
        if parse_err or data is None:
            return [], parse_err or "LLM không trả về JSON hợp lệ."

        items = data.get("questions", data if isinstance(data, list) else [])
        if not isinstance(items, list):
            return [], "JSON thiếu mảng 'questions'."

        questions: list[GeneratedQuestionItem] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            q_text = str(item.get("question", "")).strip()
            if not q_text:
                continue

            citations = enrich_citations(
                parse_citations(item.get("citations", [])), chunk_lookup
            )

            questions.append(
                GeneratedQuestionItem(
                    question=q_text,
                    question_type=str(
                        item.get("question_type", item.get("questionType", "technical"))
                    ),
                    difficulty=str(item.get("difficulty", "medium")),
                    rationale=str(item.get("rationale", "")),
                    sample_answer=str(
                        item.get("sample_answer", item.get("sampleAnswer", ""))
                    ).strip(),
                    citations=citations,
                )
            )

        if not questions:
            return [], "JSON không có câu hỏi hợp lệ."
        return questions, None

    def _parse_questions_from_plan(
        self,
        raw: str,
        chunk_lookup: dict[tuple[str, str, int], RetrievedChunk],
        approved_plan: QuestionGenerationPlan,
    ) -> tuple[list[GeneratedQuestionItem], str | None]:
        questions, parse_error = self._parse_questions(raw, chunk_lookup)
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
                        q.order = int(order_val)
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

        validation_error = _validate_questions_against_plan(questions, approved_plan)
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
) -> str | None:
    if len(questions) != plan.total_questions:
        return (
            f"Số câu hỏi ({len(questions)}) "
            f"không khớp approvedPlan.totalQuestions ({plan.total_questions})."
        )

    if plan.question_type_distribution:
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
            if sum(actual.values()) != plan.total_questions:
                return "Phân bổ questionType không khớp approved plan."

    return None
