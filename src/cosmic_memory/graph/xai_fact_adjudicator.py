"""xAI-backed internal adjudicator for ambiguous fact invalidation and merging."""

from __future__ import annotations

import asyncio
import os
import random

from cosmic_memory.graph.fact_adjudication import (
    FactAdjudicationDecision,
    FactAdjudicationRequest,
    FactAdjudicationService,
)

_SYSTEM_PROMPT = """You adjudicate ambiguous graph fact updates inside Cosmic memory.

Your job is to decide whether a newly extracted fact:
- should coexist with existing active facts,
- is the same fact and should merge into an existing relation,
- should invalidate one or more existing active facts,
- or should be discarded.

Rules:
- Be conservative. Do not invalidate existing facts unless the contradiction or supersession is strongly supported.
- Use the episode provenance and time anchors to interpret current vs historical statements.
- Prefer merge_with_existing only when the pending fact is meaningfully the same as an existing fact.
- Prefer invalidate_existing when the pending fact updates or replaces a previously active fact.
- Prefer keep_both when multiple facts can reasonably coexist.
- Prefer discard_new only if the pending fact is clearly redundant, stale, or lower-quality than an existing supported fact.
- Never invent facts, entities, or times not present in the request.
- If the pending fact is present-tense/current and the candidate fact is also active but points to a mutually exclusive state, invalidate the older one.
"""


class XAIFactAdjudicationService:
    """Structured-output fact adjudication using xAI reasoning models."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = "grok-4-1-fast-reasoning",
        timezone_name: str = "UTC",
        max_parallel_requests: int = 2,
        max_retries: int = 3,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 12.0,
        client=None,
    ) -> None:
        self.model_name = model_name
        self.timezone_name = timezone_name
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self._semaphore = asyncio.Semaphore(max(1, max_parallel_requests))
        self._owns_client = client is None
        self._client = client or self._build_client(api_key)

    async def adjudicate(self, request: FactAdjudicationRequest) -> FactAdjudicationDecision:
        async with self._semaphore:
            attempt = 0
            while True:
                try:
                    return await asyncio.to_thread(self._adjudicate_once, request)
                except Exception as exc:
                    if attempt >= self.max_retries or not _is_retryable_error(exc):
                        raise
                    delay = min(
                        self.retry_max_seconds,
                        self.retry_base_seconds * (2**attempt),
                    ) * (1.0 + (random.random() * 0.15))
                    await asyncio.sleep(delay)
                    attempt += 1

    async def close(self) -> None:
        if not self._owns_client:
            return
        close = getattr(self._client, "close", None)
        if callable(close):
            await asyncio.to_thread(close)

    def _adjudicate_once(self, request: FactAdjudicationRequest) -> FactAdjudicationDecision:
        from xai_sdk.chat import system, user

        chat = self._client.chat.create(
            model=self.model_name,
            temperature=0,
        )
        chat.append(system(_SYSTEM_PROMPT))
        chat.append(user(_build_user_prompt(request)))

        parsed = None
        if hasattr(chat, "parse"):
            parsed = chat.parse(FactAdjudicationDecision)
        else:
            sampled = chat.sample()
            parsed = getattr(sampled, "content", sampled)
        return _coerce_decision(parsed)

    @staticmethod
    def _build_client(api_key: str | None):
        try:
            from xai_sdk import Client
        except ImportError as exc:
            raise ImportError(
                "xai-sdk is required for XAIFactAdjudicationService. "
                "Install project dependencies with `python -m pip install -e .[llm]`."
            ) from exc

        resolved_api_key = api_key or os.environ.get("XAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("XAI_API_KEY is required for xAI fact adjudication.")
        return Client(api_key=resolved_api_key)


def _build_user_prompt(request: FactAdjudicationRequest) -> str:
    return f"""Adjudicate this pending graph fact against active candidates.

Time anchors:
- Current UTC time: {request.utc_time_anchor.isoformat()}
- Current local time ({request.timezone_name}): {request.local_time_anchor}
- Provenance created_at: {request.provenance_created_at.isoformat()}

Episode:
{request.episode.model_dump(mode="json")}

Pending fact:
{request.pending_fact.model_dump(mode="json")}

Candidate active facts:
{[candidate.model_dump(mode="json") for candidate in request.candidate_facts]}

Source text:
\"\"\"
{request.source_text or ""}
\"\"\"

Return one of:
- keep_both
- merge_with_existing
- invalidate_existing
- discard_new

Use chosen_relation_id only for merge_with_existing.
Use invalidated_relation_ids only for invalidate_existing.
"""


def _coerce_decision(parsed) -> FactAdjudicationDecision:
    if isinstance(parsed, FactAdjudicationDecision):
        return parsed
    if isinstance(parsed, tuple):
        for item in reversed(parsed):
            coerced = _maybe_coerce(item)
            if coerced is not None:
                return coerced
        raise ValueError("xAI parse result did not contain a FactAdjudicationDecision payload.")
    coerced = _maybe_coerce(parsed)
    if coerced is None:
        raise ValueError("Unable to coerce xAI adjudication result into FactAdjudicationDecision.")
    return coerced


def _maybe_coerce(value) -> FactAdjudicationDecision | None:
    if value is None:
        return None
    if isinstance(value, FactAdjudicationDecision):
        return value
    if hasattr(value, "model_dump"):
        return FactAdjudicationDecision.model_validate(value.model_dump(mode="python"))
    if isinstance(value, str):
        return FactAdjudicationDecision.model_validate_json(value)
    if isinstance(value, dict):
        return FactAdjudicationDecision.model_validate(value)
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return FactAdjudicationDecision.model_validate_json(content)
    if isinstance(content, dict):
        return FactAdjudicationDecision.model_validate(content)
    return None


def _is_retryable_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if response_status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    name = exc.__class__.__name__.lower()
    if any(marker in name for marker in ("ratelimit", "timeout", "connection", "apistatuserror")):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("rate limit", "temporarily unavailable", "timed out", "timeout", "connection reset")
    )
