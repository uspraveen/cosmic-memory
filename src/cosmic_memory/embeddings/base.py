"""Dense embedding service contract."""

from __future__ import annotations

from typing import Protocol

from cosmic_memory.domain.models import GenerateEmbeddingsRequest, GenerateEmbeddingsResponse


class EmbeddingService(Protocol):
    """Async contract for dense embedding generation."""

    model_name: str
    dimensions: int

    async def generate(
        self, request: GenerateEmbeddingsRequest
    ) -> GenerateEmbeddingsResponse: ...

    async def close(self) -> None: ...
