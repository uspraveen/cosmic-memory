"""xAI-backed internal adjudicator for ambiguous entity creation and merging."""

from __future__ import annotations

import asyncio
import os
import random
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cosmic_memory.graph.adjudication import (
    EntityAdjudicationDecision,
    EntityAdjudicationRequest,
)

_SYSTEM_PROMPT = """You adjudicate ambiguous entity creation inside Cosmic memory.

Your job is to decide whether a pending entity should:
- merge into one existing candidate entity,
- remain a candidate_match because the evidence is plausible but not safe,
- or be created as a new entity.

Rules:
- Never invent entities that are not in the candidate list.
- Prefer exact_match only when the pending entity and existing entity clearly refer to the same real-world thing.
- Use entity type, aliases, attributes, match reasons, relation summaries, and source text context together.
- Vector similarity is only a hint, never the final proof by itself.
- If there are multiple plausible candidates and you are not highly confident, choose candidate_match.
- If none of the candidates are strong enough, choose created_new.
- If you choose exact_match, chosen_entity_id must be one of the provided candidates.
- If you choose candidate_match, include the best candidate_entity_ids, ordered best first.
- Keep rationale short and grounded.
"""


class XAIEntityAdjudicationService:
    """Structured-output entity adjudication using xAI reasoning models."""

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

    async def adjudicate(
        self,
        request: EntityAdjudicationRequest,
    ) -> EntityAdjudicationDecision:
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

    def _adjudicate_once(
        self,
        request: EntityAdjudicationRequest,
    ) -> EntityAdjudicationDecision:
        from xai_sdk.chat import system, user

        chat = self._client.chat.create(
            model=self.model_name,
            temperature=0,
        )
        chat.append(system(_SYSTEM_PROMPT))
        chat.append(user(_build_user_prompt(request)))

        parsed = None
        if hasattr(chat, "parse"):
            parsed = chat.parse(EntityAdjudicationDecision)
        else:
            sampled = chat.sample()
            parsed = getattr(sampled, "content", sampled)
        return _coerce_adjudication_result(parsed)

    @staticmethod
    def _build_client(api_key: str | None):
        try:
            from xai_sdk import Client
        except ImportError as exc:
            raise ImportError(
                "xai-sdk is required for XAIEntityAdjudicationService. "
                "Install project dependencies with `python -m pip install -e .[llm]`."
            ) from exc

        resolved_api_key = api_key or os.environ.get("XAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("XAI_API_KEY is required for xAI entity adjudication.")
        return Client(api_key=resolved_api_key)


def _build_user_prompt(request: EntityAdjudicationRequest) -> str:
    return f"""Adjudicate an ambiguous entity write for Cosmic memory.

Time anchors:
- Current UTC time: {request.utc_time_anchor.isoformat()}
- Current local time ({request.timezone_name}): {request.local_time_anchor.isoformat()}
- Provenance created_at: {request.provenance_created_at.isoformat()}

Pending entity:
{request.pending_entity.model_dump(mode="json")}

Candidate entities:
{[candidate.model_dump(mode="json") for candidate in request.candidate_entities]}

Source text:
\"\"\"
{request.source_text or ""}
\"\"\"
"""


def _coerce_adjudication_result(parsed) -> EntityAdjudicationDecision:
    if isinstance(parsed, EntityAdjudicationDecision):
        return parsed
    if isinstance(parsed, tuple):
        for item in reversed(parsed):
            coerced = _maybe_coerce(item)
            if coerced is not None:
                return coerced
        raise ValueError("xAI parse result did not contain an EntityAdjudicationDecision payload.")
    coerced = _maybe_coerce(parsed)
    if coerced is None:
        raise ValueError("Unable to coerce xAI adjudication result into EntityAdjudicationDecision.")
    return coerced


def _maybe_coerce(value) -> EntityAdjudicationDecision | None:
    if value is None:
        return None
    if isinstance(value, EntityAdjudicationDecision):
        return value
    if hasattr(value, "model_dump"):
        return EntityAdjudicationDecision.model_validate(value.model_dump(mode="python"))
    if isinstance(value, str):
        return EntityAdjudicationDecision.model_validate_json(value)
    if isinstance(value, dict):
        return EntityAdjudicationDecision.model_validate(value)
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return EntityAdjudicationDecision.model_validate_json(content)
    if isinstance(content, dict):
        return EntityAdjudicationDecision.model_validate(content)
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


def build_local_time_anchor(timezone_name: str) -> datetime:
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        tz = timezone.utc
    return datetime.now(tz)
