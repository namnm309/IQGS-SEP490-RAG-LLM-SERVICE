"""SCRUM-457 / SCRUM-461: Adaptive competency — skill khóa theo CV, chunk lệch stack bị drop."""
from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from config.settings import Settings
from models.internal_schemas import (
    AdaptiveBlueprintRequest,
    AdaptiveBlueprintResponse,
    AdaptiveCompetencyDto,
    AdaptiveCitationDto,
    AdaptiveRoadmapRequest,
    AdaptiveRoadmapResponse,
    AdaptiveRoadmapTopicDto,
    CompetencyContextChunkDto,
    CompetencyContextRequest,
    CompetencyContextResponse,
)
from services.cv_skill_chunk_filter import (
    filter_competency_chunks,
    filter_retrieved_chunks,
    map_skill_to_allowed,
)
from services.json_output_parser import (
    build_json_fix_prompt,
    extract_json_object,
    is_retryable_json_error,
)
from services.rag_retrieval_service import RagRetrievalService
from vectorstores.base import RetrievedChunk

logger = logging.getLogger(__name__)

CATEGORIES = {"FUNDAMENTAL", "ROLE_CORE", "ADVANCED"}

# SCRUM-461: skillName phải thuộc cvSkills; chunk chỉ là evidence tùy chọn.
BLUEPRINT_SYSTEM_PROMPT = """Bạn là competency architect cho phần mềm.
Nguồn sự thật về skill là cvSkills. retrievedChunks chỉ là tài liệu tham khảo (có thể rỗng).

## Quy tắc
1. Trả JSON thuần, không markdown.
2. skillName BẮT BUỘC thuộc cvSkills (khớp tên hoặc chứa lẫn nhau). KHÔNG invent skill ngoài CV (vd. không thêm ASP.NET nếu CV chỉ React).
3. Số competencies: 1 đến min(8, số cvSkills). Σweight = 1.00 (±0.01).
4. category ∈ FUNDAMENTAL | ROLE_CORE | ADVANCED.
5. Có retrievedChunks: ưu tiên topics từ chunk; chunkIndexes optional.
6. Không có retrievedChunks: được suy topics từ CV + targetRole (inferred); không bắt chunkIndexes.
7. Không gán targetScore — Backend gán từ policy.

## Schema
{
  "competencies": [
    {
      "skillKey": "string",
      "skillName": "string",
      "category": "FUNDAMENTAL|ROLE_CORE|ADVANCED",
      "weight": 0.0,
      "topics": ["string"],
      "chunkIndexes": [0]
    }
  ]
}
"""

ROADMAP_SYSTEM_PROMPT = """Bạn là coach lộ trình kỹ năng Adaptive.
Ưu tiên topic thuộc allowedTopics / retrievedChunks của ĐÚNG skill đang luyện.
Khi retrievedChunks rỗng: được suy topic từ skill + blueprint (inferred). KHÔNG bịa skill khác stack.

## Schema
{
  "topics": [
    {"topic": "string", "subtopic": "string|null", "chunkIndex": 0, "reason": "string"}
  ],
  "explanation": "string"
}
"""


class AdaptiveCompetencyService:
    def __init__(
        self,
        retrieval: RagRetrievalService,
        client: OpenAI,
        settings: Settings,
    ) -> None:
        self._retrieval = retrieval
        self._client = client
        self._settings = settings

    def retrieve_context(self, request: CompetencyContextRequest) -> CompetencyContextResponse:
        skill_list = [s for s in request.skills if s and str(s).strip()]
        skills = " ".join(skill_list)
        query = f"{request.target_role} {request.target_level} competency skills {skills}".strip()
        raw = self._retrieve(query, skills)

        # SCRUM-461: đã retrieve nhưng lệch stack → drop hết, vẫn success + chunks=[] để BE inferred.
        if raw:
            filtered = filter_retrieved_chunks(raw, skill_list) if skill_list else raw
            if not filtered and skill_list:
                logger.warning(
                    "Adaptive retrieve: %s chunk lệch stack CV %s — trả rỗng để inferred.",
                    len(raw),
                    skill_list,
                )
                return CompetencyContextResponse(success=True, chunks=[])
            return CompetencyContextResponse(success=True, chunks=self._map_chunks(filtered))

        # Không retrieve được gì: nếu còn CV skills → BE sẽ inferred; chỉ fail khi không có skills.
        if skill_list:
            logger.warning("Adaptive EMPTY_RETRIEVAL nhưng có cvSkills — success chunks=[] (inferred).")
            return CompetencyContextResponse(success=True, chunks=[])

        return CompetencyContextResponse(
            success=False,
            error="EMPTY_RETRIEVAL: không có chunk SYSTEM Tech KB cho vai trò này.",
        )

    def generate_blueprint(self, request: AdaptiveBlueprintRequest) -> AdaptiveBlueprintResponse:
        cv_skills = [s for s in request.cv_skills if s and str(s).strip()]
        # Lọc lần nữa defense in depth (BE cũng lọc).
        chunks = filter_competency_chunks(request.chunks, cv_skills) if cv_skills else list(request.chunks)

        if not chunks and not cv_skills:
            return AdaptiveBlueprintResponse(
                success=False,
                error="EMPTY_RETRIEVAL: không sinh blueprint khi thiếu chunk và thiếu cvSkills.",
            )

        max_comps = min(8, max(1, len(cv_skills))) if cv_skills else 8
        payload = {
            "targetRole": request.target_role,
            "targetLevel": request.target_level,
            "cvSkills": cv_skills,
            "maxCompetencies": max_comps,
            "kbSource": "system" if chunks else "inferred",
            "retrievedChunks": [
                {
                    "chunkIndex": i,
                    "sourceTitle": c.source_title,
                    "sourceUrl": c.source_url,
                    "section": c.section,
                    "documentType": c.document_type,
                    "content": (c.content or "")[:1200],
                }
                for i, c in enumerate(chunks)
            ],
        }
        data, err = self._ask_json(BLUEPRINT_SYSTEM_PROMPT, payload)
        if data is None:
            return AdaptiveBlueprintResponse(success=False, error=err or "LLM blueprint failed")

        comps, verr = self._validate_blueprint(data, chunks, cv_skills)
        if verr:
            retry_payload = {
                **payload,
                "validationError": verr,
            }
            data2, err2 = self._ask_json(
                BLUEPRINT_SYSTEM_PROMPT,
                retry_payload,
                extra_user=f"Output trước sai validation: {verr}. Sửa JSON theo schema.",
            )
            if data2 is None:
                return AdaptiveBlueprintResponse(success=False, error=err2 or verr)
            comps, verr = self._validate_blueprint(data2, chunks, cv_skills)
            if verr:
                return AdaptiveBlueprintResponse(success=False, error=verr)

        return AdaptiveBlueprintResponse(success=True, competencies=comps)

    def generate_roadmap(self, request: AdaptiveRoadmapRequest) -> AdaptiveRoadmapResponse:
        skill_filter = [request.skill] if request.skill and str(request.skill).strip() else []
        chunks = (
            filter_competency_chunks(request.chunks, skill_filter)
            if skill_filter
            else list(request.chunks)
        )

        # Chỉ nhét section/title từ chunk ĐÃ lọc — không cho .NET vào allowed khi luyện React.
        allowed = {self._norm(t) for t in request.blueprint_topics if t and str(t).strip()}
        for c in chunks:
            if c.section:
                allowed.add(self._norm(c.section))
            if c.source_title:
                allowed.add(self._norm(c.source_title))

        # Inferred: không có allowed cứng — LLM được suy; vẫn ưu tiên blueprint topics.
        allow_inferred = not chunks
        if not allowed and not request.blueprint_topics and not allow_inferred:
            return AdaptiveRoadmapResponse(
                success=False,
                error="Không có topic hợp lệ (retrieved ∪ blueprint).",
            )

        payload = {
            "targetRole": request.target_role,
            "targetLevel": request.target_level,
            "skill": request.skill,
            "currentScore": request.current_score,
            "targetScore": request.target_score,
            "gap": request.gap,
            "kbSource": "system" if chunks else "inferred",
            "allowedTopics": sorted(t for t in (request.blueprint_topics or []) if t),
            "retrievedChunks": [
                {
                    "chunkIndex": i,
                    "sourceTitle": c.source_title,
                    "section": c.section,
                    "content": (c.content or "")[:800],
                }
                for i, c in enumerate(chunks)
            ],
        }
        data, err = self._ask_json(ROADMAP_SYSTEM_PROMPT, payload)
        picks: list[AdaptiveRoadmapTopicDto] = []
        explanation = None
        if data is not None:
            explanation = data.get("explanation") if isinstance(data, dict) else None
            raw_topics = data.get("topics") if isinstance(data, dict) else None
            if isinstance(raw_topics, list):
                for item in raw_topics:
                    if not isinstance(item, dict):
                        continue
                    topic = str(item.get("topic") or "").strip()
                    if not topic:
                        continue
                    # Có allowed từ blueprint/chunk: bắt buộc trong tập; inferred thuần: chấp nhận topic LLM.
                    if allowed and not allow_inferred and self._norm(topic) not in allowed:
                        continue
                    if allowed and allow_inferred and self._norm(topic) not in allowed:
                        # Vẫn chấp nhận topic suy luận ngoài allowed khi inferred.
                        pass
                    idx = item.get("chunkIndex")
                    chunk = None
                    if isinstance(idx, int) and 0 <= idx < len(chunks):
                        chunk = chunks[idx]
                    picks.append(
                        AdaptiveRoadmapTopicDto(
                            topic=topic,
                            subtopic=item.get("subtopic"),
                            source_title=chunk.source_title if chunk else None,
                            source_url=chunk.source_url if chunk else None,
                            section=chunk.section if chunk else None,
                            document_id=self._as_id(chunk.document_id) if chunk else None,
                        )
                    )

        if not picks:
            fallback = [t for t in request.blueprint_topics if t and str(t).strip()][:5]
            if not fallback and request.skill:
                skill = request.skill.strip()
                fallback = [
                    f"{skill} fundamentals",
                    f"{skill} application",
                    f"{skill} advanced",
                ]
            picks = [AdaptiveRoadmapTopicDto(topic=t) for t in fallback]
            explanation = explanation or "LLM không chọn topic hợp lệ — dùng blueprint/inferred topics."

        if not picks:
            return AdaptiveRoadmapResponse(success=False, error=err or "Không có topic roadmap.")
        return AdaptiveRoadmapResponse(success=True, topics=picks, explanation=explanation)

    def _retrieve(self, query: str, skills: str) -> list[RetrievedChunk]:
        try:
            chunks = self._retrieval.retrieve_system_only(
                query,
                query_extra=skills or None,
                metadata_filters={"documentType": "InternalStack"},
            )
            if chunks:
                return chunks
        except Exception:
            logger.warning("adaptive retrieve InternalStack failed — thử không filter")
        try:
            return self._retrieval.retrieve_system_only(query, query_extra=skills or None)
        except Exception:
            logger.exception("adaptive retrieve_system_only failed")
            return []

    def _map_chunks(self, chunks: list[RetrievedChunk]) -> list[CompetencyContextChunkDto]:
        mapped: list[CompetencyContextChunkDto] = []
        for c in chunks:
            meta = c.metadata or {}
            mapped.append(
                CompetencyContextChunkDto(
                    source_title=meta.get("sourceTitle") or meta.get("title") or meta.get("fileName"),
                    source_url=meta.get("sourceUrl") or meta.get("url"),
                    document_id=self._as_id(c.document_id),
                    section=meta.get("section") or meta.get("topic"),
                    document_type=meta.get("documentType"),
                    content=c.content or "",
                    score=c.score,
                )
            )
        return mapped

    def _validate_blueprint(
        self,
        data: Any,
        chunks: list[CompetencyContextChunkDto],
        cv_skills: list[str],
    ) -> tuple[list[AdaptiveCompetencyDto], str | None]:
        if not isinstance(data, dict):
            return [], "JSON phải là object."
        raw = data.get("competencies")
        if not isinstance(raw, list):
            return [], "Thiếu competencies[]."

        max_comps = min(8, max(1, len(cv_skills))) if cv_skills else 8
        min_comps = 1 if cv_skills else 3
        if not min_comps <= len(raw) <= max_comps:
            return [], f"Cần {min_comps}–{max_comps} competencies, nhận {len(raw)}."

        comps: list[AdaptiveCompetencyDto] = []
        weights: list[float] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                return [], "Mỗi competency phải là object."
            name_raw = str(item.get("skillName") or "").strip()
            if not name_raw:
                return [], "skillName bắt buộc."

            # SCRUM-461: khóa skill ⊆ CV.
            mapped = map_skill_to_allowed(name_raw, cv_skills) if cv_skills else name_raw
            if cv_skills and mapped is None:
                return [], f"skillName '{name_raw}' không thuộc cvSkills."
            name = mapped or name_raw
            norm_key = self._norm(name)
            if norm_key in seen:
                continue
            seen.add(norm_key)

            category = str(item.get("category") or "ROLE_CORE").strip().upper()
            if category not in CATEGORIES:
                category = "ROLE_CORE"
            try:
                weight = float(item.get("weight") or 0)
            except (TypeError, ValueError):
                return [], f"weight không hợp lệ cho {name}."
            topics = [str(t).strip() for t in (item.get("topics") or []) if str(t).strip()]
            if not topics:
                return [], f"{name} thiếu topics."

            indexes = item.get("chunkIndexes") or item.get("chunkIndex")
            if isinstance(indexes, int):
                indexes = [indexes]
            if not isinstance(indexes, list):
                indexes = []

            citations: list[AdaptiveCitationDto] = []
            for idx in indexes:
                if not isinstance(idx, int) or idx < 0 or idx >= len(chunks):
                    # Chunk optional khi inferred — bỏ index lệch, không fail cả blueprint.
                    continue
                ch = chunks[idx]
                citations.append(
                    AdaptiveCitationDto(
                        source_title=ch.source_title,
                        source_url=ch.source_url,
                        section=ch.section,
                        document_id=ch.document_id,
                        excerpt=(ch.content or "")[:240],
                    )
                )

            key = str(item.get("skillKey") or name).strip().lower().replace(" ", "-")
            comps.append(
                AdaptiveCompetencyDto(
                    skill_key=key,
                    skill_name=name,
                    category=category,
                    weight=weight,
                    topics=topics,
                    citations=citations,
                )
            )
            weights.append(weight)

        if not comps:
            return [], "Không còn competency hợp lệ sau khi khóa theo CV."
        if len(comps) > max_comps:
            return [], f"Cần tối đa {max_comps} competencies sau map CV."

        total = sum(weights)
        if abs(total - 1.0) > 0.01:
            # Renormalize nhẹ khi LLM lệch sau khi drop duplicate.
            if total > 0:
                for c in comps:
                    c.weight = round(c.weight / total, 4)
                # Fix rounding drift trên phần tử cuối.
                drift = 1.0 - sum(c.weight for c in comps)
                comps[-1].weight = round(comps[-1].weight + drift, 4)
            else:
                return [], f"Σweight={total:.3f}, phải = 1.00 ±0.01."
        return comps, None

    def _ask_json(
        self, system: str, payload: dict, extra_user: str | None = None
    ) -> tuple[Any | None, str | None]:
        user = json.dumps(payload, ensure_ascii=False)
        if extra_user:
            user = extra_user + "\n" + user
        try:
            raw = self._complete(system, user)
        except Exception:
            logger.exception("adaptive LLM call failed")
            return None, "LLM unavailable"
        data, err = extract_json_object(raw)
        if data is not None:
            return data, None
        if err and is_retryable_json_error(err):
            try:
                raw2 = self._complete(system, build_json_fix_prompt(raw))
                data2, err2 = extract_json_object(raw2)
                if data2 is not None:
                    return data2, None
                return None, err2 or err
            except Exception:
                logger.exception("adaptive JSON retry failed")
        return None, err or "LLM không trả JSON hợp lệ."

    def _complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._settings.chat_model,
            temperature=self._settings.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    @staticmethod
    def _norm(value: str) -> str:
        return (value or "").strip().lower()

    @staticmethod
    def _as_id(value: str | None) -> str | None:
        if not value:
            return None
        return str(value)
