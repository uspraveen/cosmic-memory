"""Sparse encoder implementations for tests and non-native fallbacks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol


@dataclass
class SparseVectorData:
    indices: list[int]
    values: list[float]


class SparseEncoder(Protocol):
    async def encode(self, texts: list[str]) -> list[SparseVectorData]: ...


class SimpleSparseEncoder:
    """Deterministic sparse encoder for development and tests."""

    def __init__(self, dimensions: int = 50_000) -> None:
        self.dimensions = dimensions

    async def encode(self, texts: list[str]) -> list[SparseVectorData]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> SparseVectorData:
        buckets: dict[int, float] = {}
        for token in _tokenize(text):
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            buckets[index] = buckets.get(index, 0.0) + 1.0

        indices = sorted(buckets)
        return SparseVectorData(
            indices=indices,
            values=[buckets[index] for index in indices],
        )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())
