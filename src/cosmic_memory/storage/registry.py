"""SQLite registry for canonical memory records."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from cosmic_memory.domain.enums import MemoryKind, RecordStatus
from cosmic_memory.domain.models import CanonicalMemorySnapshot, MemoryRecord


class RegistryEntry(BaseModel):
    memory_id: str
    kind: MemoryKind
    status: RecordStatus
    version: int
    path: str
    content_hash: str
    title: str | None = None
    tags: list[str]
    supersedes: str | None = None
    superseded_by: str | None = None
    created_at: datetime
    updated_at: datetime


class GraphSyncQueueEntry(BaseModel):
    job_id: int
    memory_id: str
    content_hash: str
    status: str
    attempts: int
    allow_llm: bool
    persist_graph_document: bool
    queued_at: datetime
    available_at: datetime
    updated_at: datetime
    lease_token: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_error: str | None = None


class SQLiteMemoryRegistry:
    """Fast lookup registry for canonical memory files."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def upsert(self, record: MemoryRecord, path: str | Path, content_hash: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_registry (
                    memory_id,
                    kind,
                    status,
                    version,
                    path,
                    content_hash,
                    title,
                    tags_json,
                    supersedes,
                    superseded_by,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id) DO UPDATE SET
                    kind=excluded.kind,
                    status=excluded.status,
                    version=excluded.version,
                    path=excluded.path,
                    content_hash=excluded.content_hash,
                    title=excluded.title,
                    tags_json=excluded.tags_json,
                    supersedes=excluded.supersedes,
                    superseded_by=excluded.superseded_by,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at
                """,
                (
                    record.memory_id,
                    record.kind.value,
                    record.status.value,
                    record.version,
                    str(path),
                    content_hash,
                    record.title,
                    json.dumps(record.tags, ensure_ascii=False),
                    record.supersedes,
                    record.superseded_by,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )

    def get(self, memory_id: str) -> RegistryEntry | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT memory_id, kind, status, version, path, content_hash, title, tags_json,
                       supersedes, superseded_by, created_at, updated_at
                FROM memory_registry
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_entry(row)

    def replace_all(self, snapshots: list[CanonicalMemorySnapshot]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memory_registry")
            conn.executemany(
                """
                INSERT INTO memory_registry (
                    memory_id,
                    kind,
                    status,
                    version,
                    path,
                    content_hash,
                    title,
                    tags_json,
                    supersedes,
                    superseded_by,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        snapshot.record.memory_id,
                        snapshot.record.kind.value,
                        snapshot.record.status.value,
                        snapshot.record.version,
                        snapshot.path,
                        snapshot.content_hash,
                        snapshot.record.title,
                        json.dumps(snapshot.record.tags, ensure_ascii=False),
                        snapshot.record.supersedes,
                        snapshot.record.superseded_by,
                        snapshot.record.created_at.isoformat(),
                        snapshot.record.updated_at.isoformat(),
                    )
                    for snapshot in snapshots
                ],
            )

    def delete_many(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        placeholders = ", ".join("?" for _ in memory_ids)
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM memory_registry WHERE memory_id IN ({placeholders})",
                memory_ids,
            )

    def list(
        self,
        *,
        status: RecordStatus | None = None,
        kinds: list[MemoryKind] | None = None,
    ) -> list[RegistryEntry]:
        query = """
            SELECT memory_id, kind, status, version, path, content_hash, title, tags_json,
                   supersedes, superseded_by, created_at, updated_at
            FROM memory_registry
        """
        conditions: list[str] = []
        params: list[object] = []

        if status is not None:
            conditions.append("status = ?")
            params.append(status.value)

        if kinds:
            placeholders = ", ".join("?" for _ in kinds)
            conditions.append(f"kind IN ({placeholders})")
            params.extend(kind.value for kind in kinds)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY updated_at DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_entry(row) for row in rows]

    def enqueue_graph_sync(
        self,
        *,
        memory_id: str,
        content_hash: str,
        allow_llm: bool,
        persist_graph_document: bool,
    ) -> GraphSyncQueueEntry:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            existing = conn.execute(
                """
                SELECT job_id
                FROM graph_sync_queue
                WHERE memory_id = ?
                  AND content_hash = ?
                  AND allow_llm = ?
                  AND persist_graph_document = ?
                  AND status IN ('pending', 'running', 'failed')
                ORDER BY job_id DESC
                LIMIT 1
                """,
                (
                    memory_id,
                    content_hash,
                    1 if allow_llm else 0,
                    1 if persist_graph_document else 0,
                ),
            ).fetchone()
            if existing is not None:
                row = conn.execute(
                    """
                    SELECT *
                    FROM graph_sync_queue
                    WHERE job_id = ?
                    """,
                    (existing["job_id"],),
                ).fetchone()
                return self._row_to_graph_sync_queue_entry(row)

            cursor = conn.execute(
                """
                INSERT INTO graph_sync_queue (
                    memory_id,
                    content_hash,
                    status,
                    attempts,
                    allow_llm,
                    persist_graph_document,
                    queued_at,
                    available_at,
                    updated_at,
                    lease_token,
                    started_at,
                    completed_at,
                    last_error
                ) VALUES (?, ?, 'pending', 0, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)
                """,
                (
                    memory_id,
                    content_hash,
                    1 if allow_llm else 0,
                    1 if persist_graph_document else 0,
                    now,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                """
                SELECT *
                FROM graph_sync_queue
                WHERE job_id = ?
                """,
                (int(cursor.lastrowid),),
            ).fetchone()
        return self._row_to_graph_sync_queue_entry(row)

    def lease_next_graph_sync_job(self) -> GraphSyncQueueEntry | None:
        now = datetime.now().isoformat()
        lease_token = uuid4().hex
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM graph_sync_queue
                WHERE status IN ('pending', 'failed')
                  AND available_at <= ?
                ORDER BY
                  CASE status WHEN 'pending' THEN 0 ELSE 1 END,
                  queued_at ASC,
                  job_id ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if row is None:
                return None

            updated = conn.execute(
                """
                UPDATE graph_sync_queue
                SET status = 'running',
                    attempts = attempts + 1,
                    updated_at = ?,
                    started_at = ?,
                    lease_token = ?
                WHERE job_id = ?
                  AND status IN ('pending', 'failed')
                """,
                (
                    now,
                    now,
                    lease_token,
                    row["job_id"],
                ),
            )
            if updated.rowcount != 1:
                return None

            leased = conn.execute(
                """
                SELECT *
                FROM graph_sync_queue
                WHERE job_id = ?
                """,
                (row["job_id"],),
            ).fetchone()
        return self._row_to_graph_sync_queue_entry(leased)

    def mark_graph_sync_job_succeeded(
        self,
        *,
        job_id: int,
        lease_token: str,
        status: str = "succeeded",
        last_error: str | None = None,
    ) -> None:
        if status not in {"succeeded", "stale"}:
            raise ValueError("Unsupported graph sync completion status.")
        now = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE graph_sync_queue
                SET status = ?,
                    updated_at = ?,
                    completed_at = ?,
                    lease_token = NULL,
                    last_error = ?
                WHERE job_id = ?
                  AND lease_token = ?
                """,
                (
                    status,
                    now,
                    now,
                    last_error,
                    job_id,
                    lease_token,
                ),
            )

    def mark_graph_sync_job_failed(
        self,
        *,
        job_id: int,
        lease_token: str,
        error_message: str,
        retry_delay_seconds: float,
    ) -> None:
        now = datetime.now()
        available_at = now + timedelta(seconds=max(0.0, retry_delay_seconds))
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE graph_sync_queue
                SET status = 'failed',
                    updated_at = ?,
                    available_at = ?,
                    completed_at = NULL,
                    lease_token = NULL,
                    last_error = ?
                WHERE job_id = ?
                  AND lease_token = ?
                """,
                (
                    now.isoformat(),
                    available_at.isoformat(),
                    error_message,
                    job_id,
                    lease_token,
                ),
            )

    def requeue_running_graph_sync_jobs(self) -> int:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE graph_sync_queue
                SET status = 'pending',
                    updated_at = ?,
                    available_at = ?,
                    lease_token = NULL,
                    started_at = NULL,
                    completed_at = NULL
                WHERE status = 'running'
                """,
                (
                    now,
                    now,
                ),
            )
            return int(updated.rowcount or 0)

    def graph_sync_queue_counts(self) -> dict[str, int]:
        counts = {
            "pending": 0,
            "running": 0,
            "failed": 0,
            "succeeded": 0,
            "stale": 0,
        }
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM graph_sync_queue
                GROUP BY status
                """
            ).fetchall()
        for row in rows:
            status = str(row["status"])
            if status in counts:
                counts[status] = int(row["count"] or 0)
        return counts

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_registry (
                    memory_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    title TEXT,
                    tags_json TEXT NOT NULL,
                    supersedes TEXT,
                    superseded_by TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_registry_kind_status ON memory_registry(kind, status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_registry_updated_at ON memory_registry(updated_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_sync_queue (
                    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    allow_llm INTEGER NOT NULL DEFAULT 1,
                    persist_graph_document INTEGER NOT NULL DEFAULT 1,
                    queued_at TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    lease_token TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    last_error TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_sync_queue_status_available
                ON graph_sync_queue(status, available_at, queued_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_sync_queue_memory_id
                ON graph_sync_queue(memory_id, queued_at)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> RegistryEntry:
        return RegistryEntry(
            memory_id=row["memory_id"],
            kind=MemoryKind(row["kind"]),
            status=RecordStatus(row["status"]),
            version=row["version"],
            path=row["path"],
            content_hash=row["content_hash"],
            title=row["title"],
            tags=json.loads(row["tags_json"]),
            supersedes=row["supersedes"],
            superseded_by=row["superseded_by"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_graph_sync_queue_entry(row: sqlite3.Row) -> GraphSyncQueueEntry:
        return GraphSyncQueueEntry(
            job_id=int(row["job_id"]),
            memory_id=row["memory_id"],
            content_hash=row["content_hash"],
            status=row["status"],
            attempts=int(row["attempts"] or 0),
            allow_llm=bool(row["allow_llm"]),
            persist_graph_document=bool(row["persist_graph_document"]),
            queued_at=datetime.fromisoformat(row["queued_at"]),
            available_at=datetime.fromisoformat(row["available_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            lease_token=row["lease_token"],
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
            last_error=row["last_error"],
        )
