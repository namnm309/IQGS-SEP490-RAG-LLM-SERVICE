"""Tải tài liệu từ blobReadUrl vào tempfile, tự xóa khi thoát context."""
from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import httpx

from config.settings import Settings

logger = logging.getLogger(__name__)


class DocumentDownloader:
    def __init__(self, settings: Settings):
        self._settings = settings

    @contextmanager
    def download_to_tempfile(self, blob_read_url: str, file_name: str) -> Iterator[Path]:
        suffix = Path(file_name).suffix or ".bin"
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = Path(tmp.name)

            timeout = httpx.Timeout(self._settings.request_timeout_seconds)
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                with client.stream("GET", blob_read_url) as response:
                    response.raise_for_status()
                    with temp_path.open("wb") as out:
                        for chunk in response.iter_bytes(chunk_size=65536):
                            out.write(chunk)

            logger.info(
                "Đã tải %s (%s bytes) từ %s",
                file_name,
                temp_path.stat().st_size,
                urlparse(blob_read_url).netloc,
            )
            yield temp_path
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError as exc:
                    logger.warning("Không xóa được tempfile %s: %s", temp_path, exc)
