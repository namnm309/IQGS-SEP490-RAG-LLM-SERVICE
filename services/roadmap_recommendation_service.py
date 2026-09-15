"""SCRUM-455: LLM chọn/sắp topic roadmap trong tập retrieved — không đổi score/level."""
from __future__ import annotations

import json
import logging

from openai import OpenAI

from config.settings import Settings
from models.internal_schemas import (
    RoadmapCandidateNodeDto,
    RoadmapRecommendRequest,
    RoadmapRecommendResponse,
    RoadmapTopicPickDto,
)
from services.json_output_parser import extract_json_object
from services.rag_retrieval_service import RagRetrievalService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là coach lộ trình kỹ năng. Chỉ SẮP XẾP và GIẢI THÍCH các topic đã cho.

## Quy tắc bắt buộc
1. Chỉ được chọn topic nằm trong candidateNodes (đúng chữ topic).
2. KHÔNG bịa topic mới. KHÔNG đổi currentScore / targetScore / gap / level.
3. Ưu tiên topic nền tảng (prerequisites) trước topic nâng cao.
4. Trả JSON thuần, không markdown.
5. Giải thích bằng tiếng Việt, ngắn.

## Schema
{
  "topics": [{"topic": "string", "reason": "string"}],
  "explanation": "string"
}
"""


class RoadmapRecommendationService:
    def __init__(
        self,
        retrieval: RagRetrievalService,
        client: OpenAI,
        settings: Settings,
    ) -> None:
        self._retrieval = retrieval
        self._client = client
        self._settings = settings

    def recommend(self, request: RoadmapRecommendRequest) -> RoadmapRecommendResponse:
        allowed = self._allowed_nodes(request)
        if not allowed:
            return RoadmapRecommendResponse(
                success=False,
                error="Không có candidateNodes để chọn topic.",
            )

        retrieved = self._retrieve_optional(request, allowed)
        pool = allowed
        if retrieved:
            # Ưu tiên node vừa retrieve được nhưng vẫn chỉ trong tập Backend gửi.
            retrieved_topics = {self._norm(c.metadata.get("topic")) for c in retrieved}
            retrieved_topics.discard("")
            if retrieved_topics:
                pool = [
                    n for n in allowed if self._norm(n.topic) in retrieved_topics
                ] or allowed

        try:
            picks, explanation = self._ask_llm(request, pool)
        except Exception:
            logger.exception("roadmap-recommendation LLM failed — fallback order")
            picks, explanation = [], None

        validated = self._validate_picks(picks, pool)
        if not validated:
            validated = [
                RoadmapTopicPickDto(topic=n.topic, reason="Fallback theo importance")
                for n in sorted(pool, key=lambda x: (-(x.importance or 0), x.topic))
            ]
            explanation = explanation or "LLM không chọn được topic hợp lệ — dùng thứ tự curated."

        return RoadmapRecommendResponse(
            success=True,
            topics=validated,
            explanation=explanation,
        )

    def _retrieve_optional(
        self, request: RoadmapRecommendRequest, allowed: list[RoadmapCandidateNodeDto]
    ):
        skill = (request.weak_skills[0].skill if request.weak_skills else "") or (
            allowed[0].skill if allowed else ""
        )
        # Chỉ retrieve khi BE gửi documentIds từ folder coach-roadmap — không quét full SYSTEM.
        doc_ids = [str(d).strip() for d in (request.document_ids or []) if str(d).strip()]
        if not doc_ids:
            return []

        query = f"{request.target_role} {request.target_level} {skill} roadmap topics"
        filters: dict[str, str] = {}
        if request.role_key:
            filters["roleKey"] = request.role_key
        if request.target_level:
            filters["level"] = request.target_level
        if skill:
            filters["skill"] = skill
        try:
            return self._retrieval.retrieve_system_only(
                query,
                query_extra=skill,
                document_ids=doc_ids,
                metadata_filters=filters or None,
            )
        except Exception:
            logger.warning("roadmap retrieve failed — dùng candidateNodes Backend gửi")
            return []

    def _ask_llm(
        self,
        request: RoadmapRecommendRequest,
        pool: list[RoadmapCandidateNodeDto],
    ) -> tuple[list[RoadmapTopicPickDto], str | None]:
        payload = {
            "targetRole": request.target_role,
            "targetLevel": request.target_level,
            "weakSkills": [
                s.model_dump(by_alias=True) for s in request.weak_skills
            ],
            "candidateNodes": [n.model_dump(by_alias=True) for n in pool],
            "rules": [
                "Chỉ chọn topic trong candidateNodes",
                "Không đổi score/target/level",
            ],
        }
        raw = (
            self._client.chat.completions.create(
                model=self._settings.chat_model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            .choices[0]
            .message.content
            or ""
        )
        parsed, err = extract_json_object(raw)
        if parsed is None:
            raise ValueError(err or "LLM không trả JSON")
        topics = []
        for row in parsed.get("topics") or []:
            if not isinstance(row, dict):
                continue
            topic = str(row.get("topic") or "").strip()
            if not topic:
                continue
            topics.append(
                RoadmapTopicPickDto(topic=topic, reason=row.get("reason"))
            )
        explanation = parsed.get("explanation")
        return topics, str(explanation) if explanation else None

    @staticmethod
    def _allowed_nodes(request: RoadmapRecommendRequest) -> list[RoadmapCandidateNodeDto]:
        seen: set[str] = set()
        result: list[RoadmapCandidateNodeDto] = []
        for node in request.candidate_nodes:
            key = (node.topic or "").strip()
            if not key:
                continue
            norm = key.casefold()
            if norm in seen:
                continue
            seen.add(norm)
            result.append(node)
        return result

    @staticmethod
    def _validate_picks(
        picks: list[RoadmapTopicPickDto],
        pool: list[RoadmapCandidateNodeDto],
    ) -> list[RoadmapTopicPickDto]:
        allowed = {n.topic.casefold(): n.topic for n in pool if n.topic}
        out: list[RoadmapTopicPickDto] = []
        seen: set[str] = set()
        for pick in picks:
            key = pick.topic.casefold()
            if key not in allowed or key in seen:
                continue
            seen.add(key)
            out.append(RoadmapTopicPickDto(topic=allowed[key], reason=pick.reason))
        return out

    @staticmethod
    def _norm(value: object) -> str:
        return str(value or "").strip().casefold()
