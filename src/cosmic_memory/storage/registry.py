"""SQLite registry for canonical memory records."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

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
