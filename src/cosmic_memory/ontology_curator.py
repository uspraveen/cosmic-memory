"""LLM-backed soft ontology curation for Cosmic memory."""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from cosmic_memory.domain.models import OntologyAliasItem
from cosmic_memory.graph.ontology import EntityType, RelationType
from cosmic_memory.usage import (
    GatewayUsageLogger,
    begin_metered_call,
    extract_provider_request_id,
    extract_usage_payload,
)

_SYSTEM_PROMPT = """You curate soft ontology growth inside Cosmic memory.

Your job is to review recurring weak-fit ontology observations from the write path.

The live graph schema is still fixed. You may only:
- map a recurring alias_label to one existing ontology type,
- decide to keep observing because evidence is not strong enough,
- or mark the concept as propose_new because the current ontology is genuinely a poor fit.

Rules:
- Be conservative.
- Prefer mapping to an existing type when retrieval and graph behavior would still be coherent.
- Only choose map_to_existing when the alias_label clearly refers to the same recurring concept family.
- Use the observed examples, fallback types, fit levels, and recurrence counts together.
- If evidence is noisy, one-off, or underspecified, choose keep_observing.
- Choose propose_new only when repeated evidence strongly suggests the current ontology is a poor fit.
- mapped_type must be one of the provided allowed types when decision=map_to_existing.
- Never invent a mapped_type that is not in the allowed list.
- Keep rationale short and grounded.
"""


class OntologyObservationGroup(BaseModel):
    observation_kind: Literal["entity_type", "relation_type"]
    alias_label: str
    observation_count: int = 0
    fit_counts: dict[str, int] = Field(default_factory=dict)
    fallback_type_counts: dict[str, int] = Field(default_factory=dict)
    example_evidence: list[str] = Field(default_factory=list)
    example_rationales: list[str] = Field(default_factory=list)
    example_memory_ids: list[str] = Field(default_factory=list)


class OntologyCurationDecision(BaseModel):
    decision: Literal["map_to_existing", "keep_observing", "propose_new"]
    mapped_type: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str | None = None


class OntologyCuratorService(Protocol):
    model_name: str

    async def curate_group(
        self,
        group: OntologyObservationGroup,
        *,
        learned_aliases: Sequence[OntologyAliasItem] = (),
    ) -> OntologyCurationDecision: ...

    async def close(self) -> None: ...


class XAIOntologyCuratorService:
    """Structured-output curation of recurring ontology weak-fit observations."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = "grok-4-1-fast-reasoning",
        max_parallel_requests: int = 1,
        max_retries: int = 3,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 12.0,
        usage_logger: GatewayUsageLogger | None = None,
        client=None,
    ) -> None:
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self._usage_logger = usage_logger
        self._semaphore = asyncio.Semaphore(max(1, max_parallel_requests))
        self._owns_client = client is None
        self._client = client or self._build_client(api_key)

    async def curate_group(
        self,
        group: OntologyObservationGroup,
        *,
        learned_aliases: Sequence[OntologyAliasItem] = (),
    ) -> OntologyCurationDecision:
        async with self._semaphore:
            attempt = 0
            while True:
                metered_call = begin_metered_call(prefix="call")
                try:
                    result, raw_response = await asyncio.to_thread(
                        self._curate_once,
                        group,
                        list(learned_aliases),
                    )
                    await self._emit_usage(
                        metered_call=metered_call,
                        group=group,
                        raw_response=raw_response,
                        success=True,
                        error_code=None,
                        metadata_json={"attempt": attempt + 1},
                    )
                    return result
                except Exception as exc:
                    await self._emit_usage(
                        metered_call=metered_call,
                        group=group,
                        raw_response=None,
                        success=False,
                        error_code=type(exc).__name__,
                        metadata_json={"attempt": attempt + 1},
                    )
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

    def _curate_once(
        self,
        group: OntologyObservationGroup,
        learned_aliases: list[OntologyAliasItem],
    ) -> tuple[OntologyCurationDecision, object]:
        from xai_sdk.chat import system, user

        chat = self._client.chat.create(
            model=self.model_name,
            temperature=0,
        )
        chat.append(system(_SYSTEM_PROMPT))
        chat.append(user(_build_user_prompt(group, learned_aliases=learned_aliases)))

        raw_response = None
        if hasattr(chat, "parse"):
            raw_response = chat.parse(OntologyCurationDecision)
        else:
            sampled = chat.sample()
            raw_response = getattr(sampled, "content", sampled)
        return _coerce_decision(raw_response), raw_response

    @staticmethod
    def _build_client(api_key: str | None):
        try:
            from xai_sdk import Client
        except ImportError as exc:
            raise ImportError(
                "xai-sdk is required for XAIOntologyCuratorService. "
                "Install project dependencies with `python -m pip install -e .[llm]`."
            ) from exc

        resolved_api_key = api_key or os.environ.get("XAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("XAI_API_KEY is required for xAI ontology curation.")
        return Client(api_key=resolved_api_key)

    async def _emit_usage(
        self,
        *,
        metered_call,
        group: OntologyObservationGroup,
        raw_response: object,
        success: bool,
        error_code: str | None,
        metadata_json: dict[str, object] | None,
    ) -> None:
        if self._usage_logger is None:
            return
        try:
            await self._usage_logger.emit(
                metered_call=metered_call,
                provider="xai",
                model=self.model_name,
                usage_kind="chat_completion",
                operation="memory.ontology_curate",
                raw_usage=extract_usage_payload(raw_response),
                provider_request_id=extract_provider_request_id(raw_response),
                metadata_json={
                    "observation_kind": group.observation_kind,
                    "alias_label": group.alias_label,
                    "observation_count": group.observation_count,
                    **(metadata_json or {}),
                },
                success=success,
                error_code=error_code,
            )
        except Exception:
            return


def _build_user_prompt(
    group: OntologyObservationGroup,
    *,
    learned_aliases: Sequence[OntologyAliasItem],
) -> str:
    if group.observation_kind == "entity_type":
        allowed_types = [entity_type.value for entity_type in EntityType]
        active_aliases = [
            alias.model_dump(mode="json")
            for alias in learned_aliases
            if alias.observation_kind == "entity_type"
        ]
    else:
        allowed_types = [relation_type.value for relation_type in RelationType]
        active_aliases = [
            alias.model_dump(mode="json")
            for alias in learned_aliases
            if alias.observation_kind == "relation_type"
        ]
    return f"""Curate one recurring ontology weak-fit group for Cosmic memory.

Observation kind:
{group.observation_kind}

Allowed mapped types:
{allowed_types}

Already learned aliases of the same kind:
{active_aliases}

Recurring observation group:
{group.model_dump(mode="json")}

Return one of:
- map_to_existing
- keep_observing
- propose_new

Use mapped_type only when decision=map_to_existing.
"""


def _coerce_decision(parsed) -> OntologyCurationDecision:
    if isinstance(parsed, OntologyCurationDecision):
        return parsed
    if isinstance(parsed, tuple):
        for item in reversed(parsed):
            coerced = _maybe_coerce(item)
            if coerced is not None:
                return coerced
        raise ValueError("xAI parse result did not contain an OntologyCurationDecision payload.")
    coerced = _maybe_coerce(parsed)
    if coerced is None:
        raise ValueError("Unable to coerce xAI curation result into OntologyCurationDecision.")
    return coerced


def _maybe_coerce(value) -> OntologyCurationDecision | None:
    if value is None:
        return None
    if isinstance(value, OntologyCurationDecision):
        return value
    if hasattr(value, "model_dump"):
        return OntologyCurationDecision.model_validate(value.model_dump(mode="python"))
    if isinstance(value, str):
        return OntologyCurationDecision.model_validate_json(value)
    if isinstance(value, dict):
        return OntologyCurationDecision.model_validate(value)
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return OntologyCurationDecision.model_validate_json(content)
    if isinstance(content, dict):
        return OntologyCurationDecision.model_validate(content)
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
