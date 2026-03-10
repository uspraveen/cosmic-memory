"""Markdown-backed canonical memory record store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

from cosmic_memory.domain.models import CanonicalMemorySnapshot, MemoryRecord
from cosmic_memory.storage.layout import KIND_DIRECTORY_MAP, path_for_record


class MarkdownWriteResult(BaseModel):
    path: Path
    rendered: str
    content_hash: str


class MarkdownRecordStore:
    """Stores one canonical memory record per Markdown file."""

    def __init__(self, memory_root: str | Path) -> None:
        self.memory_root = Path(memory_root)
        self.memory_root.mkdir(parents=True, exist_ok=True)
        for directory in KIND_DIRECTORY_MAP.values():
            (self.memory_root / directory).mkdir(parents=True, exist_ok=True)

    def write(self, record: MemoryRecord) -> MarkdownWriteResult:
        path = path_for_record(self.memory_root, record.memory_id, record.kind)
        rendered = render_record_markdown(record)
        content_hash = hash_markdown_content(rendered)

        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(rendered, encoding="utf-8")
        temp_path.replace(path)

        return MarkdownWriteResult(
            path=path,
            rendered=rendered,
            content_hash=content_hash,
        )

    def read(self, path: str | Path) -> MemoryRecord:
        text = Path(path).read_text(encoding="utf-8")
        return self.parse(text)

    def render(self, record: MemoryRecord) -> str:
        return render_record_markdown(record)

    def parse(self, text: str) -> MemoryRecord:
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip() != "---":
            raise ValueError("Invalid canonical memory Markdown: missing frontmatter start")

        frontmatter: dict[str, object] = {}
        end_index = None
        for index in range(1, len(lines)):
            line = lines[index]
            if line.strip() == "---":
                end_index = index
                break
            key, _, raw_value = line.partition(":")
            if not _:
                raise ValueError(f"Invalid frontmatter line: {line}")
            frontmatter[key.strip()] = json.loads(raw_value.strip())

        if end_index is None:
            raise ValueError("Invalid canonical memory Markdown: missing frontmatter end")

        body = "\n".join(lines[end_index + 1 :]).rstrip("\n")
        frontmatter["content"] = body
        return MemoryRecord.model_validate(frontmatter)

    def scan(self) -> list[CanonicalMemorySnapshot]:
        snapshots: list[CanonicalMemorySnapshot] = []
        for directory in KIND_DIRECTORY_MAP.values():
            for path in sorted((self.memory_root / directory).glob("*.md")):
                text = path.read_text(encoding="utf-8")
                record = self.parse(text)
                snapshots.append(
                    CanonicalMemorySnapshot(
                        memory_id=record.memory_id,
                        kind=record.kind,
                        status=record.status,
                        version=record.version,
                        path=str(path),
                        content_hash=hash_markdown_content(text),
                        token_count=approx_token_count(record.content),
                        record=record,
                    )
                )
        snapshots.sort(key=lambda snapshot: snapshot.record.updated_at, reverse=True)
        return snapshots


def render_record_markdown(record: MemoryRecord) -> str:
    payload = record.model_dump(mode="json", exclude={"content"})
    lines = ["---"]
    for key, value in payload.items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    lines.append("---")
    lines.append(record.content.rstrip("\n"))
    lines.append("")
    return "\n".join(lines)


def hash_markdown_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_record_hash(record: MemoryRecord) -> str:
    return hash_markdown_content(render_record_markdown(record))


def approx_token_count(text: str) -> int:
    return max(len(text.split()), 1)
