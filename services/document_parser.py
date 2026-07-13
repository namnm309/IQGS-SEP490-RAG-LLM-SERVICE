"""Parse PDF, DOCX, TXT — trích từ logic ingestion.py cũ."""
from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})


class DocumentParser:
    def supported_extension(self, file_name: str) -> bool:
        return Path(file_name).suffix.lower() in SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path, file_name: str | None = None) -> str:
        name = file_name or file_path.name
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Định dạng '{ext}' không hỗ trợ. Hỗ trợ: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if ext == ".pdf":
            text = self._extract_pdf(file_path)
        elif ext == ".docx":
            text = self._extract_docx(file_path)
        else:
            text = file_path.read_text(encoding="utf-8")

        if not text or not text.strip():
            raise ValueError("Không trích xuất được nội dung từ tài liệu")

        return text.strip()

    def _extract_pdf(self, file_path: Path) -> str:
        lines: list[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    lines.append(page_text)
        return "\n\n".join(lines)

    def _extract_docx(self, file_path: Path) -> str:
        doc = DocxDocument(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
