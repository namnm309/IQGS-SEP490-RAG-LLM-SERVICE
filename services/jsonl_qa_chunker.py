"""SCRUM-448/455: JSONL → 1 record = 1 chunk (Q/A dataset hoặc roadmap node)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QaChunkDraft:
    """Một chunk sẵn sàng embed — kèm metadata Q/A hoặc roadmap."""

    content: str
    record_index: int
    repo: str | None = None
    record_kind: str = "qa"
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class JsonlQaChunker:
    """Đọc file .jsonl: Q/A (question+answer) hoặc roadmap (kind=roadmap / topic+skill)."""

    REQUIRED_QA_KEYS = ("question", "answer")

    def build_chunks(self, file_path: Path, file_name: str | None = None) -> list[QaChunkDraft]:
        name = file_name or file_path.name
        if Path(name).suffix.lower() != ".jsonl":
            raise ValueError(f"JsonlQaChunker chỉ hỗ trợ .jsonl, nhận: {name}")

        stem = Path(name).stem
        # default.jsonl = toàn bộ dataset — không gắn repo cụ thể
        repo = None if stem.lower() == "default" else stem

        try:
            raw = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise ValueError(f"Không đọc được file JSONL: {exc}") from exc

        drafts: list[QaChunkDraft] = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Dòng {line_no}: JSON không hợp lệ — {exc.msg}"
                ) from exc

            if not isinstance(row, dict):
                raise ValueError(f"Dòng {line_no}: mỗi dòng phải là object JSON")

            if self._is_roadmap_row(row):
                drafts.append(self._roadmap_draft(row, line_no, len(drafts), repo))
                continue

            missing = [k for k in self.REQUIRED_QA_KEYS if k not in row or row[k] is None]
            if missing:
                raise ValueError(
                    f"Dòng {line_no}: thiếu field bắt buộc: {', '.join(missing)}"
                )

            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            if not question or not answer:
                raise ValueError(
                    f"Dòng {line_no}: question và answer không được để trống"
                )

            content = f"Question: {question}\nAnswer: {answer}"
            drafts.append(
                QaChunkDraft(
                    content=content,
                    record_index=len(drafts),
                    repo=repo,
                    record_kind="qa",
                )
            )

        if not drafts:
            raise ValueError("File JSONL không có record Q/A hợp lệ")

        return drafts

    @staticmethod
    def _is_roadmap_row(row: dict[str, Any]) -> bool:
        kind = str(row.get("kind") or row.get("recordKind") or "").strip().lower()
        if kind == "roadmap":
            return True
        has_topic_skill = bool(row.get("topic")) and bool(row.get("skill"))
        has_qa = "question" in row or "answer" in row
        return has_topic_skill and not has_qa

    @staticmethod
    def _roadmap_draft(
        row: dict[str, Any], line_no: int, index: int, repo: str | None
    ) -> QaChunkDraft:
        topic = str(row.get("topic") or "").strip()
        skill = str(row.get("skill") or "").strip()
        if not topic or not skill:
            raise ValueError(f"Dòng {line_no}: roadmap cần topic và skill")

        role_key = str(row.get("roleKey") or row.get("role_key") or "").strip()
        level = str(row.get("level") or "").strip()
        technology = str(row.get("technology") or "").strip()
        subtopic = str(row.get("subtopic") or "").strip()
        content_body = str(row.get("content") or "").strip()
        parts = [f"Skill: {skill}", f"Topic: {topic}"]
        if subtopic:
            parts.append(f"Subtopic: {subtopic}")
        if role_key:
            parts.append(f"Role: {role_key}")
        if level:
            parts.append(f"Level: {level}")
        if content_body:
            parts.append(content_body)
        extra: dict[str, Any] = {
            "documentType": "Roadmap",
            "roleKey": role_key or None,
            "technology": technology or None,
            "level": level or None,
            "skill": skill,
            "topic": topic,
            "subtopic": subtopic or None,
            "prerequisites": row.get("prerequisites") or [],
            "nextTopics": row.get("nextTopics") or row.get("next_topics") or [],
            "importance": row.get("importance"),
            "sourceVersion": row.get("sourceVersion") or row.get("source_version"),
            "sourceTitle": row.get("sourceTitle") or row.get("source_title"),
            "sourceUrl": row.get("sourceUrl") or row.get("source_url"),
        }
        extra = {k: v for k, v in extra.items() if v is not None and v != []}
        return QaChunkDraft(
            content="\n".join(parts),
            record_index=index,
            repo=repo,
            record_kind="roadmap",
            extra_metadata=extra,
        )
