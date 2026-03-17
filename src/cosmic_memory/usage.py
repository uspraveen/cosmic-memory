from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


_DEFAULT_TOKEN_FIELD_MAP = {
    "prompt_tokens": ["prompt_tokens", "input_tokens"],
    "completion_tokens": ["completion_tokens", "output_tokens"],
    "total_tokens": ["total_tokens"],
    "cached_tokens": ["cached_tokens", "cache_read_input_tokens"],
    "reasoning_tokens": ["reasoning_tokens"],
}

_MODEL_TOKEN_FIELD_MAP = {
    ("perplexity", "pplx-embed-v1-4b"): {
        "prompt_tokens": ["prompt_tokens", "input_tokens"],
        "completion_tokens": [],
        "total_tokens": ["total_tokens"],
        "cached_tokens": [],
        "reasoning_tokens": [],
    },
    ("xai", "grok-4-1-fast-reasoning"): {
        "prompt_tokens": ["prompt_tokens", "prompt_text_tokens", "input_tokens"],
        "completion_tokens": ["completion_tokens", "output_tokens"],
        "total_tokens": ["total_tokens"],
        "cached_tokens": ["cached_prompt_text_tokens", "cached_tokens"],
        "reasoning_tokens": ["reasoning_tokens"],
    },
}

_PROVIDER_TOKEN_FIELD_MAP = {
    "xai": _MODEL_TOKEN_FIELD_MAP[("xai", "grok-4-1-fast-reasoning")],
    "perplexity": _DEFAULT_TOKEN_FIELD_MAP,
}


@dataclass(frozen=True, slots=True)
class MeteredCall:
    llm_call_id: str
    llm_call_placed_at: str
    started_perf_counter: float


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def begin_metered_call(*, prefix: str = "call") -> MeteredCall:
    normalized_prefix = str(prefix or "call").strip().lower() or "call"
    return MeteredCall(
        llm_call_id=f"{normalized_prefix}_{uuid4().hex}",
        llm_call_placed_at=utcnow_iso(),
        started_perf_counter=time.perf_counter(),
    )


class GatewayUsageLogger:
    def __init__(
        self,
        *,
        gateway_url: str = "",
        internal_token: str = "",
        source_component: str = "session_manager",
        source_id: str = "cosmic-memory",
        timeout_sec: float = 2.5,
        max_attempts: int = 2,
        retry_base_seconds: float = 0.15,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        self.internal_token = internal_token.strip()
        self.source_component = source_component.strip() or "session_manager"
        self.source_id = source_id.strip() or "cosmic-memory"
        self.timeout_sec = max(0.25, timeout_sec)
        self.max_attempts = max(1, max_attempts)
        self.retry_base_seconds = max(0.05, retry_base_seconds)
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_sec, connect=min(2.0, self.timeout_sec)))
        self._owns_client = client is None

    @property
    def enabled(self) -> bool:
        return bool(self.gateway_url and self.internal_token)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def emit(
        self,
        *,
        metered_call: MeteredCall,
        provider: str,
        model: str,
        usage_kind: str,
        operation: str,
        raw_usage: Any = None,
        provider_request_id: str | None = None,
        task_id: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        route: str | None = None,
        user_id: str | None = None,
        success: bool = True,
        error_code: str | None = None,
        latency_ms: int | None = None,
        estimated_cost_usd: float | None = None,
        metadata_json: Any | None = None,
    ) -> bool:
        if not self.enabled:
            return False
        payload = build_usage_event(
            metered_call=metered_call,
            source_component=self.source_component,
            source_id=self.source_id,
            provider=provider,
            model=model,
            usage_kind=usage_kind,
            operation=operation,
            raw_usage=raw_usage,
            provider_request_id=provider_request_id,
            task_id=task_id,
            session_id=session_id,
            request_id=request_id,
            route=route,
            user_id=user_id,
            success=success,
            error_code=error_code,
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost_usd,
            metadata_json=metadata_json,
        )
        return await post_usage_event(
            client=self._client,
            gateway_url=self.gateway_url,
            internal_token=self.internal_token,
            event=payload,
            timeout_sec=self.timeout_sec,
            max_attempts=self.max_attempts,
            retry_base_seconds=self.retry_base_seconds,
        )


def build_usage_event(
    *,
    metered_call: MeteredCall,
    source_component: str,
    source_id: str,
    provider: str,
    model: str,
    usage_kind: str,
    operation: str,
    raw_usage: Any = None,
    provider_request_id: str | None = None,
    task_id: str | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    route: str | None = None,
    user_id: str | None = None,
    success: bool = True,
    error_code: str | None = None,
    latency_ms: int | None = None,
    estimated_cost_usd: float | None = None,
    metadata_json: Any | None = None,
) -> dict[str, Any]:
    normalized_usage = normalize_usage(provider=provider, model=model, raw_usage=raw_usage)
    if latency_ms is None:
        latency_ms = max(0, int((time.perf_counter() - metered_call.started_perf_counter) * 1000))
    if estimated_cost_usd is None:
        estimated_cost_usd = extract_provider_cost_usd(raw_usage)
    combined_metadata = merge_metadata(
        metadata_json,
        {
            "raw_usage": serialize_usage_metadata(raw_usage),
        },
    )
    return {
        "llm_call_id": metered_call.llm_call_id,
        "user_id": normalize_optional_text(user_id),
        "source_component": normalize_required_text(source_component),
        "source_id": normalize_optional_text(source_id),
        "task_id": normalize_optional_text(task_id),
        "plan_id": None,
        "parent_task_id": None,
        "session_id": normalize_optional_text(session_id),
        "route": normalize_optional_text(route),
        "operation": normalize_required_text(operation),
        "usage_kind": normalize_required_text(usage_kind),
        "provider": normalize_required_text(provider),
        "model": normalize_required_text(model),
        "request_id": normalize_optional_text(request_id),
        "provider_request_id": normalize_optional_text(provider_request_id),
        "prompt_tokens": normalized_usage["prompt_tokens"],
        "completion_tokens": normalized_usage["completion_tokens"],
        "total_tokens": normalized_usage["total_tokens"],
        "cached_tokens": normalized_usage["cached_tokens"],
        "reasoning_tokens": normalized_usage["reasoning_tokens"],
        "estimated_cost_usd": estimated_cost_usd,
        "latency_ms": latency_ms,
        "success": bool(success),
        "error_code": normalize_optional_text(error_code) if not success else None,
        "metadata_json": combined_metadata,
        "llm_call_placed_at": metered_call.llm_call_placed_at,
    }


async def post_usage_event(
    *,
    client: httpx.AsyncClient,
    gateway_url: str,
    internal_token: str,
    event: dict[str, Any],
    timeout_sec: float = 2.5,
    max_attempts: int = 2,
    retry_base_seconds: float = 0.15,
) -> bool:
    if not gateway_url.strip() or not internal_token.strip():
        return False
    url = gateway_url.rstrip("/") + "/internal/usage/log"
    headers = {
        "Content-Type": "application/json",
        "X-Internal-Token": internal_token,
    }
    for attempt in range(max(1, max_attempts)):
        try:
            response = await client.post(
                url,
                headers=headers,
                json=event,
                timeout=httpx.Timeout(timeout_sec, connect=min(timeout_sec, 1.5)),
            )
            if response.status_code in {200, 201}:
                logger.debug(
                    "cosmic_memory.usage_logged llm_call_id=%s operation=%s provider=%s model=%s",
                    event.get("llm_call_id"),
                    event.get("operation"),
                    event.get("provider"),
                    event.get("model"),
                )
                return True
            response.raise_for_status()
            return True
        except Exception:
            if attempt >= max(1, max_attempts) - 1:
                logger.warning(
                    "cosmic_memory.usage_log_failed llm_call_id=%s operation=%s provider=%s model=%s",
                    event.get("llm_call_id"),
                    event.get("operation"),
                    event.get("provider"),
                    event.get("model"),
                    exc_info=True,
                )
                return False
            await asyncio.sleep(retry_base_seconds * (2**attempt))
    return False


def normalize_usage(*, provider: str, model: str, raw_usage: Any) -> dict[str, int]:
    token_field_map = (
        _MODEL_TOKEN_FIELD_MAP.get((provider, model))
        or _PROVIDER_TOKEN_FIELD_MAP.get(provider)
        or _DEFAULT_TOKEN_FIELD_MAP
    )
    prompt_tokens = read_first_int(raw_usage, token_field_map.get("prompt_tokens", []))
    completion_tokens = read_first_int(raw_usage, token_field_map.get("completion_tokens", []))
    total_tokens = read_first_int(raw_usage, token_field_map.get("total_tokens", []))
    cached_tokens = read_first_int(raw_usage, token_field_map.get("cached_tokens", []))
    reasoning_tokens = read_first_int(raw_usage, token_field_map.get("reasoning_tokens", []))

    if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
        total_tokens = prompt_tokens + completion_tokens
    if cached_tokens > prompt_tokens:
        cached_tokens = prompt_tokens
    if reasoning_tokens > completion_tokens:
        reasoning_tokens = completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


def extract_usage_payload(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        usage = value.get("usage")
        if usage is not None:
            return usage
        nested = value.get("response")
        if nested is not None:
            return extract_usage_payload(nested)
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            usage = extract_usage_payload(item)
            if usage is not None:
                return usage
        return None
    usage_attr = getattr(value, "usage", None)
    if usage_attr is not None:
        return usage_attr
    for attr_name in ("response", "_response"):
        nested = getattr(value, attr_name, None)
        if nested is not None:
            usage = extract_usage_payload(nested)
            if usage is not None:
                return usage
    return None


def extract_provider_request_id(value: Any) -> str | None:
    candidate = _extract_request_id(value)
    return normalize_optional_text(candidate)


def extract_provider_cost_usd(raw_usage: Any) -> float | None:
    for candidate in (
        "cost.total_cost",
        "cost.total_cost_usd",
        "cost_usd",
        "total_cost_usd",
        "total_cost",
    ):
        value = read_path(raw_usage, candidate)
        try:
            if value is None or value == "":
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def serialize_usage_metadata(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): serialize_usage_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_usage_metadata(item) for item in value]
    if hasattr(value, "model_dump"):
        return serialize_usage_metadata(value.model_dump(mode="json"))
    if hasattr(value, "dict") and callable(value.dict):
        return serialize_usage_metadata(value.dict())
    if hasattr(value, "__dict__"):
        return serialize_usage_metadata(vars(value))
    return str(value)


def merge_metadata(primary: Any | None, secondary: dict[str, Any]) -> Any | None:
    secondary_clean = {key: value for key, value in secondary.items() if value is not None}
    if primary is None:
        return secondary_clean or None
    if isinstance(primary, dict):
        return {
            **primary,
            **secondary_clean,
        }
    if not secondary_clean:
        return primary
    return {
        "value": primary,
        **secondary_clean,
    }


def normalize_required_text(value: Any) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("value must be non-empty")
    return normalized


def normalize_optional_text(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def read_first_int(raw_usage: Any, candidates: list[str]) -> int:
    for candidate in candidates:
        value = read_path(raw_usage, candidate)
        if value is None or value == "":
            continue
        try:
            return max(0, int(float(value)))
        except (TypeError, ValueError):
            continue
    return 0


def read_path(value: Any, path: str) -> Any:
    if value is None or not path:
        return None
    current = value
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
            continue
        if hasattr(current, part):
            current = getattr(current, part)
            continue
        return None
    return current


def _extract_request_id(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("provider_request_id", "request_id", "response_id", "id"):
            candidate = value.get(key)
            if candidate:
                return candidate
        nested = value.get("response")
        if nested is not None:
            return _extract_request_id(nested)
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            candidate = _extract_request_id(item)
            if candidate:
                return candidate
        return None
    for attr_name in ("provider_request_id", "request_id", "response_id", "id"):
        candidate = getattr(value, attr_name, None)
        if candidate:
            return candidate
    for attr_name in ("response", "_response"):
        nested = getattr(value, attr_name, None)
        if nested is not None:
            candidate = _extract_request_id(nested)
            if candidate:
                return candidate
    return None
