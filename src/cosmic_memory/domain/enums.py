"""Core enums for cosmic-memory."""

from enum import StrEnum


class MemoryKind(StrEnum):
    CORE_FACT = "core_fact"
    SESSION_SUMMARY = "session_summary"
    TASK_SUMMARY = "task_summary"
    AGENT_NOTE = "agent_note"
    USER_DATA = "user_data"
    TRANSCRIPT = "transcript"


class RecordStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


class CoreFactConfirmationStatus(StrEnum):
    CONFIRMED = "confirmed"
    UNCONFIRMED = "unconfirmed"
    CONTESTED = "contested"
