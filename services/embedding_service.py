"""Embedding batch qua OpenAI-compatible client (Ollama), validate dimension."""
from __future__ import annotations

import logging

from openai import OpenAI

from config.settings import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, client: OpenAI, settings: Settings):
        self._client = client
        self._model = settings.embedding_model
        self._dimension = settings.embedding_dimension
        self._batch_size = max(1, settings.embedding_batch_size)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size

        for batch_index in range(total_batches):
            start = batch_index * self._batch_size
            batch = texts[start : start + self._batch_size]
            logger.info(
                "Embedding batch %s/%s (%s texts)",
                batch_index + 1,
                total_batches,
                len(batch),
            )

            response = self._client.embeddings.create(model=self._model, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            batch_embeddings = [item.embedding for item in ordered]

            for idx, embedding in enumerate(batch_embeddings):
                if len(embedding) != self._dimension:
                    global_index = start + idx
                    raise ValueError(
                        f"Embedding index {global_index} có dimension {len(embedding)}, "
                        f"kỳ vọng {self._dimension}"
                    )

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        results = self.embed_texts([text])
        return results[0]
