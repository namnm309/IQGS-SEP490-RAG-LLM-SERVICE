"""Tests cho json_output_parser."""
from __future__ import annotations

import json

from services.json_output_parser import (
    build_json_fix_prompt,
    extract_json_object,
    strip_markdown_fences,
)


def test_strip_markdown_fences_json() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert strip_markdown_fences(raw) == '{"a": 1}'


def test_strip_markdown_fences_plain() -> None:
    raw = '{"a": 1}'
    assert strip_markdown_fences(raw) == '{"a": 1}'


def test_extract_json_object_clean() -> None:
    data, err = extract_json_object('{"questions": []}')
    assert err is None
    assert data == {"questions": []}


def test_extract_json_object_with_extra_text() -> None:
    raw = 'Here is the result:\n{"questions": [{"q": 1}]}\nDone.'
    data, err = extract_json_object(raw)
    assert err is None
    assert data["questions"][0]["q"] == 1


def test_extract_json_object_invalid() -> None:
    data, err = extract_json_object("not json at all")
    assert data is None
    assert err is not None
    assert "JSON" in err


def test_extract_json_invalid_escape_repaired() -> None:
    # LLM chèn \s (invalid) trong string
    raw = r'{"questions":[{"question":"path C:\sers\x","question_type":"technical","difficulty":"medium","rationale":"a","sample_answer":"b","citations":[]}]}'
    data, err = extract_json_object(raw)
    assert err is None
    assert data is not None
    assert "questions" in data


def test_build_json_fix_prompt_truncates() -> None:
    prompt = build_json_fix_prompt("x" * 5000, max_len=100)
    assert len(prompt) < 500


def test_extract_json_from_fenced_block() -> None:
    payload = {"plan": {"totalQuestions": 5}}
    raw = f"```json\n{json.dumps(payload)}\n```"
    data, err = extract_json_object(raw)
    assert err is None
    assert data["plan"]["totalQuestions"] == 5
