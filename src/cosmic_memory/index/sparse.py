"""Sparse encoder implementations for tests and non-native fallbacks."""

from __future__ import annotations

import asyncio
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


class FastEmbedSparseEncoder:
    """FastEmbed-backed sparse encoder for local-path Qdrant deployments."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        try:
            from fastembed import SparseTextEmbedding
        except ImportError as exc:
            raise ImportError(
                "fastembed is required for FastEmbedSparseEncoder. "
                "Install project dependencies with `python -m pip install -e .[qdrant-local]`."
            ) from exc

        self.model_name = model_name
        self._model = SparseTextEmbedding(model_name=model_name)
        self._lock = asyncio.Lock()

    async def encode(self, texts: list[str]) -> list[SparseVectorData]:
        async with self._lock:
            embeddings = await asyncio.to_thread(lambda: list(self._model.embed(texts)))
        return [_coerce_sparse_vector(embedding) for embedding in embeddings]


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


def _coerce_sparse_vector(raw_embedding) -> SparseVectorData:
    indices = getattr(raw_embedding, "indices", None)
    values = getattr(raw_embedding, "values", None)
    if indices is None or values is None:
        raise TypeError("FastEmbed sparse embedding is missing indices or values.")
    return SparseVectorData(
        indices=[int(value) for value in list(indices)],
        values=[float(value) for value in list(values)],
    )
