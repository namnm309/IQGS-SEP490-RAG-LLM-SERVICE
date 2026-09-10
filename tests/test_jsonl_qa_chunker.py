"""SCRUM-448: unit tests cho JsonlQaChunker."""
from __future__ import annotations

from pathlib import Path

import pytest

from services.jsonl_qa_chunker import JsonlQaChunker


def _write(tmp: Path, name: str, lines: list[str]) -> Path:
    path = tmp / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_build_chunks_swe_qa_style(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "flask.jsonl",
        [
            '{"question": "What is ensure_ascii?", "answer": "Controls Unicode escaping."}',
            '{"question": "Where is it defined?", "answer": "DefaultJSONProvider."}',
        ],
    )
    drafts = JsonlQaChunker().build_chunks(path, "flask.jsonl")
    assert len(drafts) == 2
    assert drafts[0].repo == "flask"
    assert drafts[0].content.startswith("Question: What is ensure_ascii?")
    assert "Answer: Controls Unicode escaping." in drafts[0].content
    assert drafts[1].record_index == 1


def test_default_jsonl_has_no_repo(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "default.jsonl",
        ['{"question": "Q1", "answer": "A1"}'],
    )
    drafts = JsonlQaChunker().build_chunks(path, "default.jsonl")
    assert len(drafts) == 1
    assert drafts[0].repo is None


def test_rejects_missing_fields(tmp_path: Path) -> None:
    path = _write(tmp_path, "bad.jsonl", ['{"question": "only Q"}'])
    with pytest.raises(ValueError, match="thiếu field"):
        JsonlQaChunker().build_chunks(path, "bad.jsonl")


def test_rejects_invalid_json(tmp_path: Path) -> None:
    path = _write(tmp_path, "bad.jsonl", ["not-json"])
    with pytest.raises(ValueError, match="JSON không hợp lệ"):
        JsonlQaChunker().build_chunks(path, "bad.jsonl")


def test_rejects_empty_file(tmp_path: Path) -> None:
    path = _write(tmp_path, "empty.jsonl", ["", "  "])
    with pytest.raises(ValueError, match="không có record"):
        JsonlQaChunker().build_chunks(path, "empty.jsonl")


def test_build_chunks_roadmap_kind(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "dotnet-backend.jsonl",
        [
            '{"kind":"roadmap","roleKey":"dotnet-backend","level":"Junior","skill":"C#",'
            '"topic":"async/await pitfalls","content":"deadlock pitfalls","importance":0.8}'
        ],
    )
    drafts = JsonlQaChunker().build_chunks(path, "dotnet-backend.jsonl")
    assert len(drafts) == 1
    assert drafts[0].record_kind == "roadmap"
    assert "async/await pitfalls" in drafts[0].content
    assert drafts[0].extra_metadata["documentType"] == "Roadmap"
    assert drafts[0].extra_metadata["roleKey"] == "dotnet-backend"
    assert drafts[0].extra_metadata["skill"] == "C#"
