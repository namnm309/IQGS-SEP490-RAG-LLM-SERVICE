"""
TextChunker — tương đương RecursiveCharacterTextSplitter của LangChain.

Thuật toán chia đệ quy:
  1. Thử chia theo "\n\n" (đoạn văn)
  2. Nếu vẫn quá dài → chia theo "\n" (dòng)
  3. Nếu vẫn quá dài → chia theo ". " (câu)
  4. Nếu vẫn quá dài → chia theo kích thước cố định
"""

import re
from typing import List

SEPARATORS = ["\n\n", "\n", ". ", " "]


def split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Chia văn bản thành các chunks có kích thước chunk_size,
    với phần overlap giữa các chunks liền kề.
    """
    if not text or not text.strip():
        return []

    text = re.sub(r'\r\n|\r', '\n', text)  # normalize line endings

    if len(text) <= chunk_size:
        return [text.strip()]

    chunks: List[str] = []
    _split_recursive(text, chunk_size, overlap, SEPARATORS, 0, chunks)
    return chunks


def _split_recursive(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: List[str],
    sep_index: int,
    result: List[str],
) -> None:
    # Đủ nhỏ → thêm vào kết quả
    if len(text) <= chunk_size:
        stripped = text.strip()
        if stripped:
            result.append(stripped)
        return

    # Hết separator → chia cứng
    if sep_index >= len(separators):
        result.extend(_hard_split(text, chunk_size, overlap))
        return

    sep = separators[sep_index]
    parts = [p for p in text.split(sep) if p]

    if len(parts) <= 1:
        # Separator này không có → thử tiếp
        _split_recursive(text, chunk_size, overlap, separators, sep_index + 1, result)
        return

    current = ""
    for part in parts:
        candidate = part if not current else current + sep + part

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunk_text = current.strip()
                if len(chunk_text) <= chunk_size:
                    if chunk_text:
                        result.append(chunk_text)
                else:
                    _split_recursive(chunk_text, chunk_size, overlap, separators, sep_index + 1, result)

            # Overlap từ chunk trước
            overlap_text = ""
            if result and overlap > 0:
                last = result[-1]
                overlap_text = last[-overlap:] + sep if len(last) > overlap else last + sep

            current = overlap_text + part

    # Phần còn lại
    if current:
        remaining = current.strip()
        if remaining:
            if len(remaining) <= chunk_size:
                result.append(remaining)
            else:
                _split_recursive(remaining, chunk_size, overlap, separators, sep_index + 1, result)


def _hard_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks
