"""Deterministic development embeddings used when no remote provider is configured."""

from __future__ import annotations

import math
import re
from hashlib import sha256

from cosmic_memory.domain.models import (
    EmbeddingItem,
    GenerateEmbeddingsRequest,
    GenerateEmbeddingsResponse,
)


class HashEmbeddingService:
    """Deterministic local embedder for development and tests."""

    def __init__(self, *, dimensions: int = 1024, model_name: str = "hash-embedding-dev") -> None:
        self.dimensions = dimensions
        self.model_name = model_name

    async def generate(self, request: GenerateEmbeddingsRequest) -> GenerateEmbeddingsResponse:
        dimensions = request.dimensions or self.dimensions
        items = [
            EmbeddingItem(
                index=index,
                vector=self._embed_one(text, dimensions=dimensions, normalize=request.normalize),
                dimensions=dimensions,
            )
            for index, text in enumerate(request.texts)
        ]
        return GenerateEmbeddingsResponse(
            model=self.model_name,
            dimensions=dimensions,
            items=items,
        )

    async def close(self) -> None:
        return None

    def _embed_one(self, text: str, *, dimensions: int, normalize: bool) -> list[float]:
        vector = [0.0] * dimensions
        for token in _tokenize(text):
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        if not normalize:
            return vector

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())
