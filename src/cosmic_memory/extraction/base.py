"""Graph extraction service contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from cosmic_memory.domain.models import MemoryRecord

if TYPE_CHECKING:
    from cosmic_memory.extraction.models import GraphExtractionResult


class GraphExtractionService(Protocol):
    model_name: str

    async def extract(self, record: MemoryRecord) -> "GraphExtractionResult | None": ...

    async def close(self) -> None: ...
