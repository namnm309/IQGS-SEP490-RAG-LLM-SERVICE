"""Tests cho streaming blob download."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.settings import Settings
from services.document_downloader import DocumentDownloader


@pytest.fixture
def settings() -> Settings:
    return Settings(
        DATABASE_URL="postgresql://postgres:postgres@localhost:5432/iqgs",
        BACKEND_INTERNAL_BASE_URL="http://localhost:5000",
        INTERNAL_API_KEY="test-secret",
        EMBEDDING_DIMENSION=768,
        REQUEST_TIMEOUT_SECONDS=30,
    )


def test_download_streams_chunks_to_tempfile(settings: Settings, tmp_path: Path) -> None:
    downloader = DocumentDownloader(settings)
    payload = b"chunk-one" + b"chunk-two" + b"chunk-three"

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_bytes = MagicMock(return_value=iter([b"chunk-one", b"chunk-two", b"chunk-three"]))

    @contextmanager
    def fake_stream(method: str, url: str):
        assert method == "GET"
        assert url == "http://example.com/file.pdf"
        yield mock_response

    mock_client = MagicMock()
    mock_client.stream = fake_stream

    @contextmanager
    def fake_client(*args, **kwargs):
        yield mock_client

    with patch("services.document_downloader.httpx.Client", fake_client):
        with downloader.download_to_tempfile("http://example.com/file.pdf", "file.pdf") as path:
            assert path.exists()
            assert path.read_bytes() == payload
            temp_name = path.name

    assert not Path(temp_name).exists()
