"""Wrapper cho helpers/text_chunker.split_text."""
from config.settings import Settings
from helpers.text_chunker import split_text


class ChunkingService:
    def __init__(self, settings: Settings):
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

    def split(self, text: str) -> list[str]:
        return split_text(text, self._chunk_size, self._chunk_overlap)
