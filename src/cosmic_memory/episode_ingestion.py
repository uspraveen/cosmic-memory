"""Helpers for turning runtime observations into canonical episode memories."""

from __future__ import annotations

from cosmic_memory.domain.models import (
    EpisodeIngestResponse,
    EpisodeObservation,
    IngestEpisodeRequest,
    MemoryRecord,
    WriteMemoryRequest,
)


def build_episode_write_request(request: IngestEpisodeRequest) -> WriteMemoryRequest:
    metadata = dict(request.metadata)
    metadata["episode_observations"] = [
        _observation_payload(observation)
        for observation in request.observations
    ]
    metadata["episode_observation_count"] = len(request.observations)
    metadata["episode_type"] = request.episode_type or metadata.get("episode_type") or "observation"
    if request.extract_graph:
        metadata["extract_graph"] = True
    title = request.title or _default_episode_title(request.observations)
    return WriteMemoryRequest(
        kind=request.kind,
        title=title,
        content=render_episode_content(request.observations),
        tags=request.tags,
        metadata=metadata,
        provenance=request.provenance,
    )


def build_episode_ingest_response(record: MemoryRecord, *, observation_count: int) -> EpisodeIngestResponse:
    graph_episode_id = None
    graph_document = record.metadata.get("graph_document")
    if isinstance(graph_document, dict):
        episode_payload = graph_document.get("episode")
        if isinstance(episode_payload, dict):
            graph_episode_id = episode_payload.get("episode_id")
    return EpisodeIngestResponse(
        record=record,
        observation_count=observation_count,
        graph_episode_id=graph_episode_id,
    )


def render_episode_content(observations: list[EpisodeObservation]) -> str:
    lines: list[str] = []
    for observation in observations:
        prefix_parts = [observation.role.strip() or "unknown"]
        if observation.name:
            prefix_parts.append(observation.name.strip())
        if observation.created_at is not None:
            prefix_parts.append(observation.created_at.isoformat())
        prefix = " | ".join(prefix_parts)
        lines.append(f"[{prefix}]")
        lines.append(observation.content.strip())
        lines.append("")
    return "\n".join(lines).rstrip()


def _default_episode_title(observations: list[EpisodeObservation]) -> str:
    first = observations[0]
    snippet = " ".join(first.content.split())
    if len(snippet) > 72:
        snippet = f"{snippet[:69].rstrip()}..."
    role = first.role.strip() or "observation"
    return f"{role}: {snippet}" if snippet else role


def _observation_payload(observation: EpisodeObservation) -> dict:
    payload = observation.model_dump(mode="json")
    if payload.get("name") in {"", None}:
        payload.pop("name", None)
    if payload.get("created_at") is None:
        payload.pop("created_at", None)
    if not payload.get("metadata"):
        payload.pop("metadata", None)
    return payload
