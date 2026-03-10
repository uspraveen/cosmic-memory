"""Minimal local .env loader for production-style app factories."""

from __future__ import annotations

import os
from pathlib import Path

_LOADED_PATHS: set[Path] = set()


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> None:
    resolved = _resolve_env_path(path)
    if resolved is None or resolved in _LOADED_PATHS:
        return

    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_wrapping_quotes(value.strip())
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value

    _LOADED_PATHS.add(resolved)


def _resolve_env_path(path: str | Path | None) -> Path | None:
    configured = path or os.environ.get("COSMIC_MEMORY_ENV_FILE", ".env")
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if not candidate.exists():
        return None
    return candidate


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
