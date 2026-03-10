"""Deterministic identity normalization and key generation."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from uuid import UUID, uuid5

from cosmic_memory.domain.models import utc_now
from cosmic_memory.graph.models import GraphIdentityCandidate, GraphIdentityKey
from cosmic_memory.graph.ontology import IdentityKeyType

PERSON_EMAIL_NAMESPACE = UUID("c1f9576d-9d56-4b19-a3df-ec8713a95e11")
PERSON_PHONE_NAMESPACE = UUID("72db7ec2-8d8d-4736-9fd5-7c55f4bd29f5")
PERSON_ACCOUNT_NAMESPACE = UUID("f37c726f-0ca2-4137-b383-8df988047150")
PERSON_USERNAME_NAMESPACE = UUID("f1fe9070-4f3c-4203-b60f-5662a47dbe55")
PERSON_NAME_ALIAS_NAMESPACE = UUID("9e33aa0d-7000-43d1-8820-a67183b6f4f7")

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
HONORIFICS_PATTERN = re.compile(r"\b(dr|mr|mrs|ms|prof)\.?\b", re.IGNORECASE)


def build_identity_key(
    candidate: GraphIdentityCandidate,
    *,
    memory_id: str | None = None,
    observed_at: datetime | None = None,
) -> GraphIdentityKey:
    normalized_value = normalize_identity_value(
        candidate.key_type,
        candidate.raw_value,
        provider=candidate.provider,
    )
    key_id = deterministic_identity_key_id(
        candidate.key_type,
        normalized_value,
        provider=candidate.provider,
    )
    timestamp = observed_at or utc_now()
    memory_ids = [memory_id] if memory_id else []
    return GraphIdentityKey(
        key_id=key_id,
        key_type=candidate.key_type,
        normalized_value=normalized_value,
        raw_values=[candidate.raw_value],
        provider=(candidate.provider.casefold() if candidate.provider else None),
        confidence=candidate.confidence,
        first_seen_at=timestamp,
        last_seen_at=timestamp,
        memory_ids=memory_ids,
    )


def deterministic_identity_key_id(
    key_type: IdentityKeyType,
    normalized_value: str,
    *,
    provider: str | None = None,
) -> str:
    namespace = namespace_for_key_type(key_type)
    payload = normalized_value
    if provider:
        payload = f"{provider.casefold()}::{normalized_value}"
    return str(uuid5(namespace, payload))


def namespace_for_key_type(key_type: IdentityKeyType) -> UUID:
    return {
        IdentityKeyType.EMAIL: PERSON_EMAIL_NAMESPACE,
        IdentityKeyType.PHONE: PERSON_PHONE_NAMESPACE,
        IdentityKeyType.EXTERNAL_ACCOUNT: PERSON_ACCOUNT_NAMESPACE,
        IdentityKeyType.USERNAME: PERSON_USERNAME_NAMESPACE,
        IdentityKeyType.NAME_VARIANT: PERSON_NAME_ALIAS_NAMESPACE,
    }[key_type]


def normalize_identity_value(
    key_type: IdentityKeyType,
    raw_value: str,
    *,
    provider: str | None = None,
) -> str:
    if key_type == IdentityKeyType.EMAIL:
        return normalize_email(raw_value)
    if key_type == IdentityKeyType.PHONE:
        return normalize_phone(raw_value)
    if key_type == IdentityKeyType.EXTERNAL_ACCOUNT:
        return normalize_external_account(raw_value, provider=provider)
    if key_type == IdentityKeyType.USERNAME:
        return normalize_username(raw_value, provider=provider)
    if key_type == IdentityKeyType.NAME_VARIANT:
        return normalize_name_variant(raw_value)
    raise ValueError(f"Unsupported identity key type: {key_type}")


def normalize_email(raw_value: str) -> str:
    email = _normalize_text(raw_value)
    if not EMAIL_PATTERN.match(email):
        raise ValueError(f"Invalid email value: {raw_value}")
    local_part, domain = email.split("@", 1)
    domain = domain.casefold()
    local_part = local_part.casefold()
    if domain in {"gmail.com", "googlemail.com"}:
        local_part = local_part.split("+", 1)[0].replace(".", "")
        domain = "gmail.com"
    return f"{local_part}@{domain}"


def normalize_phone(raw_value: str) -> str:
    raw = _normalize_text(raw_value)
    has_plus = raw.startswith("+")
    digits = re.sub(r"\D", "", raw)
    if not digits:
        raise ValueError(f"Invalid phone value: {raw_value}")
    return f"+{digits}" if has_plus else digits


def normalize_external_account(raw_value: str, *, provider: str | None = None) -> str:
    value = _normalize_text(raw_value).casefold()
    if not provider:
        return value
    return f"{provider.casefold()}:{value}"


def normalize_username(raw_value: str, *, provider: str | None = None) -> str:
    value = _normalize_text(raw_value).casefold()
    if not provider:
        return value
    return f"{provider.casefold()}:{value}"


def normalize_name_variant(raw_value: str) -> str:
    value = _normalize_text(raw_value)
    value = HONORIFICS_PATTERN.sub(" ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value.casefold())
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise ValueError(f"Invalid name variant: {raw_value}")
    return value


def _normalize_text(raw_value: str) -> str:
    normalized = unicodedata.normalize("NFKC", raw_value)
    normalized = normalized.strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized
