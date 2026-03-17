"""Perplexity-backed standard embeddings for Cosmic passive recall."""

from __future__ import annotations

import asyncio
import base64
import math
import os
import random
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cosmic_memory.domain.models import (
    EmbeddingCost,
    EmbeddingItem,
    EmbeddingUsage,
    GenerateEmbeddingsRequest,
    GenerateEmbeddingsResponse,
)
from cosmic_memory.usage import (
    GatewayUsageLogger,
    begin_metered_call,
    extract_provider_request_id,
)


@dataclass(slots=True)
class _BatchEmbeddingResult:
    items: list[EmbeddingItem]
    usage: EmbeddingUsage | None


class PerplexityStandardEmbeddingService:
    """Async batch embedding service using Perplexity standard embeddings."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = "pplx-embed-v1-4b",
        dimensions: int = 1024,
        batch_size: int = 128,
        max_parallel_requests: int = 4,
        encoding_format: str = "base64_int8",
        normalize: bool = True,
        timeout: float | None = 30.0,
        max_retries: int = 4,
        retry_base_seconds: float = 0.75,
        retry_max_seconds: float = 8.0,
        usage_logger: GatewayUsageLogger | None = None,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_parallel_requests = max_parallel_requests
        self.encoding_format = encoding_format
        self.normalize = normalize
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self._usage_logger = usage_logger
        self._owns_client = client is None
        self._client = client or self._build_client(api_key)

    async def generate(self, request: GenerateEmbeddingsRequest) -> GenerateEmbeddingsResponse:
        dimensions = request.dimensions or self.dimensions
        batch_size = min(request.batch_size, 512)
        max_parallel = max(1, request.max_parallel_requests)
        encoding_format = request.encoding_format or self.encoding_format
        normalize = request.normalize

        semaphore = asyncio.Semaphore(max_parallel)
        tasks = [
            asyncio.create_task(
                self._run_batch(
                    offset=offset,
                    texts=texts,
                    dimensions=dimensions,
                    encoding_format=encoding_format,
                    normalize=normalize,
                    semaphore=semaphore,
                )
            )
            for offset, texts in _chunked_with_offsets(request.texts, batch_size)
        ]
        batch_results = await asyncio.gather(*tasks)

        items: list[EmbeddingItem] = []
        usages: list[EmbeddingUsage] = []
        for batch in batch_results:
            items.extend(batch.items)
            if batch.usage is not None:
                usages.append(batch.usage)

        items.sort(key=lambda item: item.index)
        return GenerateEmbeddingsResponse(
            model=self.model_name,
            dimensions=dimensions,
            items=items,
            usage=_merge_usage(usages),
        )

    async def close(self) -> None:
        if not self._owns_client:
            return

        close = getattr(self._client, "close", None)
        if close is not None:
            await close()

    async def _run_batch(
        self,
        *,
        offset: int,
        texts: list[str],
        dimensions: int,
        encoding_format: str,
        normalize: bool,
        semaphore: asyncio.Semaphore,
    ) -> _BatchEmbeddingResult:
        async with semaphore:
            kwargs: dict[str, Any] = {
                "input": texts,
                "model": self.model_name,
                "dimensions": dimensions,
                "encoding_format": encoding_format,
            }
            if self.timeout is not None:
                kwargs["timeout"] = self.timeout

            response = await self._create_with_retry(
                usage_metadata={
                    "text_count": len(texts),
                    "dimensions": dimensions,
                    "encoding_format": encoding_format,
                    "normalize": normalize,
                    "batch_offset": offset,
                },
                **kwargs,
            )

        items: list[EmbeddingItem] = []
        for position, entry in enumerate(response.data or []):
            encoded = getattr(entry, "embedding", None)
            if encoded is None:
                raise ValueError("Perplexity embeddings response contained a null embedding payload.")
            local_index = getattr(entry, "index", None)
            global_index = offset + (local_index if local_index is not None else position)
            vector = _decode_embedding(
                encoded,
                dimensions=dimensions,
                encoding_format=encoding_format,
                normalize=normalize,
            )
            items.append(
                EmbeddingItem(
                    index=global_index,
                    vector=vector,
                    dimensions=dimensions,
                )
            )

        return _BatchEmbeddingResult(
            items=items,
            usage=_parse_usage(getattr(response, "usage", None)),
        )

    async def _create_with_retry(self, *, usage_metadata: dict[str, Any] | None = None, **kwargs):
        attempt = 0
        while True:
            metered_call = begin_metered_call(prefix="call")
            try:
                response = await self._client.embeddings.create(**kwargs)
                await self._emit_usage(
                    metered_call=metered_call,
                    response=response,
                    success=True,
                    error_code=None,
                    metadata_json={
                        **(usage_metadata or {}),
                        "attempt": attempt + 1,
                    },
                )
                return response
            except Exception as exc:
                await self._emit_usage(
                    metered_call=metered_call,
                    response=None,
                    success=False,
                    error_code=type(exc).__name__,
                    metadata_json={
                        **(usage_metadata or {}),
                        "attempt": attempt + 1,
                    },
                )
                if attempt >= self.max_retries or not _is_retryable_error(exc):
                    raise
                delay = min(
                    self.retry_max_seconds,
                    self.retry_base_seconds * (2**attempt),
                ) * (1.0 + (random.random() * 0.15))
                await asyncio.sleep(delay)
                attempt += 1

    @staticmethod
    def _build_client(api_key: str | None):
        try:
            from perplexity import AsyncPerplexity
        except ImportError as exc:
            raise ImportError(
                "perplexityai is required for PerplexityStandardEmbeddingService. "
                "Install cosmic-memory[perplexity]."
            ) from exc

        resolved_api_key = api_key or os.environ.get("PERPLEXITY_API_KEY") or os.environ.get(
            "PPLX_API_KEY"
        )
        if not resolved_api_key:
            raise ValueError("PERPLEXITY_API_KEY is required for Perplexity embeddings.")
        return AsyncPerplexity(api_key=resolved_api_key)

    async def _emit_usage(
        self,
        *,
        metered_call,
        response: Any,
        success: bool,
        error_code: str | None,
        metadata_json: dict[str, Any] | None,
    ) -> None:
        if self._usage_logger is None:
            return
        try:
            await self._usage_logger.emit(
                metered_call=metered_call,
                provider="perplexity",
                model=self.model_name,
                usage_kind="embedding",
                operation="memory.embed",
                raw_usage=getattr(response, "usage", None),
                provider_request_id=extract_provider_request_id(response),
                success=success,
                error_code=error_code,
                metadata_json=metadata_json,
            )
        except Exception:
            return


def _chunked_with_offsets(texts: Sequence[str], batch_size: int) -> list[tuple[int, list[str]]]:
    return [
        (offset, list(texts[offset : offset + batch_size]))
        for offset in range(0, len(texts), batch_size)
    ]


def _decode_embedding(
    encoded: str,
    *,
    dimensions: int,
    encoding_format: str,
    normalize: bool,
) -> list[float]:
    raw = base64.b64decode(encoded)
    if encoding_format == "base64_int8":
        values = array("b")
        values.frombytes(raw)
        vector = [float(value) for value in values]
    elif encoding_format == "base64_binary":
        vector = _decode_binary_vector(raw, dimensions)
    else:
        raise ValueError(f"Unsupported embedding encoding format: {encoding_format}")

    if len(vector) != dimensions:
        raise ValueError(
            f"Decoded embedding dimension mismatch. Expected {dimensions}, received {len(vector)}."
        )

    if not normalize:
        return vector
    return _l2_normalize(vector)


def _decode_binary_vector(raw: bytes, dimensions: int) -> list[float]:
    vector: list[float] = []
    for byte in raw:
        for shift in range(7, -1, -1):
            vector.append(1.0 if byte & (1 << shift) else 0.0)
            if len(vector) == dimensions:
                return vector
    return vector


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _parse_usage(usage: Any) -> EmbeddingUsage | None:
    if usage is None:
        return None

    cost = getattr(usage, "cost", None)
    return EmbeddingUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
        cost=EmbeddingCost(
            currency=getattr(cost, "currency", None),
            input_cost=getattr(cost, "input_cost", None),
            total_cost=getattr(cost, "total_cost", None),
        )
        if cost is not None
        else None,
    )


def _merge_usage(usages: list[EmbeddingUsage]) -> EmbeddingUsage | None:
    if not usages:
        return None

    currency = next(
        (
            usage.cost.currency
            for usage in usages
            if usage.cost is not None and usage.cost.currency is not None
        ),
        None,
    )
    input_cost = _sum_optional(
        usage.cost.input_cost for usage in usages if usage.cost is not None
    )
    total_cost = _sum_optional(
        usage.cost.total_cost for usage in usages if usage.cost is not None
    )

    return EmbeddingUsage(
        prompt_tokens=_sum_optional(usage.prompt_tokens for usage in usages),
        total_tokens=_sum_optional(usage.total_tokens for usage in usages),
        cost=EmbeddingCost(
            currency=currency,
            input_cost=input_cost,
            total_cost=total_cost,
        )
        if currency is not None or input_cost is not None or total_cost is not None
        else None,
    )


def _sum_optional(values) -> int | float | None:
    total = 0
    found = False
    for value in values:
        if value is None:
            continue
        total += value
        found = True
    return total if found else None


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if response_status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True

    name = exc.__class__.__name__.lower()
    if any(
        marker in name
        for marker in (
            "ratelimit",
            "timeout",
            "connection",
            "internalserver",
            "apistatuserror",
        )
    ):
        return True

    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "rate limit",
            "temporarily unavailable",
            "timed out",
            "timeout",
            "connection reset",
        )
    )
