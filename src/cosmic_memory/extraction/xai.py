"""xAI-backed graph extraction for write-time memory ingestion."""

from __future__ import annotations

import asyncio
import os
import random
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cosmic_memory.domain.models import MemoryRecord
from cosmic_memory.extraction.models import GraphExtractionResult
from cosmic_memory.graph.ontology import EntityType, RelationType

_SYSTEM_PROMPT = """You extract structured graph memory for Cosmic.

You must only return grounded information from the provided memory.

Rules:
- Use only the provided ontology entity types and relation types.
- Prefer fewer, higher-quality entities and relations over broad extraction.
- Deduplicate internally. If two names clearly refer to the same thing in this memory, emit one entity.
- Use canonical display names and put alternate spellings or titles in alias_values.
- For people, emit identity candidates when the text contains grounded emails, usernames, phone numbers, or external handles.
- For first-person statements that clearly describe Cosmic's primary user, create a person entity using the provided primary-user display name when available, otherwise use "Primary User". Include one identity candidate:
  - key_type: external_account
  - provider: cosmic
  - raw_value: primary_user
- For relative time expressions such as today, yesterday, last week, tomorrow, next Friday, or currently, resolve them against the provided local and UTC time anchors.
- Use absolute ISO 8601 datetimes when grounded.
- If a relation is explicitly current, active, blocked, ongoing, or happening today/right now and no more precise timestamp is given, set valid_at to the provided Provenance created_at anchor.
- Use attended for grounded education attendance or study affiliation with a school, college, or university.
- Use graduated_from for grounded graduation or degree-completion facts tied to an institution.
- Do not use part_of for education facts when attended or graduated_from is a better fit.
- Only emit both attended and graduated_from when the memory explicitly supports both as distinct facts. If the memory only says someone graduated, prefer graduated_from alone.
- If time is implied but not precise, prefer null over guessing.
- valid_at means when a fact becomes true.
- invalid_at means when a fact stops being true.
- expires_at means when a transient fact naturally expires.
- If the memory has no meaningful graph structure, return should_extract=false.
- Do not invent entities, times, or relations that are not supported by the memory.
"""


class XAIGraphExtractionService:
    """Structured-output graph extraction using xAI reasoning models."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str = "grok-4-1-fast-reasoning",
        timezone_name: str = "UTC",
        primary_user_display_name: str | None = None,
        max_parallel_requests: int = 2,
        max_retries: int = 3,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 12.0,
        client=None,
    ) -> None:
        self.model_name = model_name
        self.timezone_name = timezone_name
        self.primary_user_display_name = (primary_user_display_name or "").strip() or None
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self._semaphore = asyncio.Semaphore(max(1, max_parallel_requests))
        self._owns_client = client is None
        self._client = client or self._build_client(api_key)

    async def extract(self, record: MemoryRecord) -> GraphExtractionResult | None:
        if not record.content.strip():
            return None

        async with self._semaphore:
            attempt = 0
            while True:
                try:
                    return await asyncio.to_thread(self._extract_once, record)
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

    def _extract_once(self, record: MemoryRecord) -> GraphExtractionResult:
        from xai_sdk.chat import system, user

        chat = self._client.chat.create(
            model=self.model_name,
            temperature=0,
        )
        chat.append(system(_SYSTEM_PROMPT))
        chat.append(
            user(
                _build_user_prompt(
                    record,
                    timezone_name=self.timezone_name,
                    primary_user_display_name=self.primary_user_display_name,
                )
            )
        )

        parsed = None
        if hasattr(chat, "parse"):
            parsed = chat.parse(GraphExtractionResult)
        else:
            sampled = chat.sample()
            parsed = getattr(sampled, "content", sampled)
        return _coerce_extraction_result(parsed)

    @staticmethod
    def _build_client(api_key: str | None):
        try:
            from xai_sdk import Client
        except ImportError as exc:
            raise ImportError(
                "xai-sdk is required for XAIGraphExtractionService. "
                "Install project dependencies with `python -m pip install -e .[llm]`."
            ) from exc

        resolved_api_key = api_key or os.environ.get("XAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("XAI_API_KEY is required for xAI graph extraction.")
        return Client(api_key=resolved_api_key)


def _build_user_prompt(
    record: MemoryRecord,
    *,
    timezone_name: str,
    primary_user_display_name: str | None = None,
) -> str:
    local_now = _local_now(timezone_name)
    utc_now = datetime.now(timezone.utc)
    tags = ", ".join(record.tags) if record.tags else "(none)"
    title = record.title or "(none)"
    provenance = record.provenance.model_dump(mode="json")
    primary_user_name = (primary_user_display_name or "").strip() or "Primary User"
    ontology = {
        "entity_types": [entity_type.value for entity_type in EntityType],
        "relation_types": [relation_type.value for relation_type in RelationType],
    }
    return f"""Extract graph memory from this canonical Cosmic memory.

Time anchors:
- Current UTC time: {utc_now.isoformat()}
- Current local time ({timezone_name}): {local_now.isoformat()}
- Memory created_at: {record.created_at.isoformat()}
- Provenance created_at: {record.provenance.created_at.isoformat()}

Memory metadata:
- memory_id: {record.memory_id}
- kind: {record.kind.value}
- title: {title}
- tags: {tags}
- provenance: {provenance}
- primary_user_display_name: {primary_user_name}

Ontology:
{ontology}

Memory content:
\"\"\"
{record.content}
\"\"\"
"""


def _coerce_extraction_result(parsed) -> GraphExtractionResult:
    if isinstance(parsed, GraphExtractionResult):
        return parsed
    if isinstance(parsed, tuple):
        for item in reversed(parsed):
            coerced = _maybe_coerce(item)
            if coerced is not None:
                return coerced
        raise ValueError("xAI parse result did not contain a GraphExtractionResult payload.")
    coerced = _maybe_coerce(parsed)
    if coerced is None:
        raise ValueError("Unable to coerce xAI extraction result into GraphExtractionResult.")
    return coerced


def _maybe_coerce(value) -> GraphExtractionResult | None:
    if value is None:
        return None
    if isinstance(value, GraphExtractionResult):
        return value
    if hasattr(value, "model_dump"):
        return GraphExtractionResult.model_validate(value.model_dump(mode="python"))
    if isinstance(value, str):
        return GraphExtractionResult.model_validate_json(value)
    if isinstance(value, dict):
        return GraphExtractionResult.model_validate(value)
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return GraphExtractionResult.model_validate_json(content)
    if isinstance(content, dict):
        return GraphExtractionResult.model_validate(content)
    return None


def _local_now(timezone_name: str) -> datetime:
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        tz = timezone.utc
    return datetime.now(tz)


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
