"""Shared retrieval helpers for memory services."""

from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from time import perf_counter

from cosmic_memory.domain.enums import MemoryKind, RecordStatus
from cosmic_memory.domain.models import (
    ActiveRecallResponse,
    ActiveRecallDiagnostics,
    GraphEntity,
    GraphEpisodeItem,
    GraphRelation,
    MemoryRecord,
    PassiveRecallDiagnostics,
    PassiveRecallResponse,
    RecallItem,
)
from cosmic_memory.graph.identity import normalize_email
from cosmic_memory.graph.models import GraphQueryFrame, GraphSearchResult
from cosmic_memory.graph.query import build_query_frame
from cosmic_memory.graph.search_recipes import GraphRecipeApplication, apply_graph_search_recipe


@dataclass(slots=True)
class QuerySignals:
    query: str
    query_lower: str
    query_tokens: set[str]
    query_frame: GraphQueryFrame
    normalized_emails: set[str]
    current_state_hint: bool
    relationship_hint: bool
    temporal_hint: bool
    task_hint: bool
    identity_hint: bool


def build_query_signals(query: str) -> QuerySignals:
    frame = build_query_frame(query)
    normalized_emails: set[str] = set()
    for candidate in frame.identity_candidates:
        if candidate.key_type.value != "email":
            continue
        try:
            normalized_emails.add(normalize_email(candidate.raw_value))
        except ValueError:
            continue

    query_lower = query.lower()
    intent_values = {intent.value for intent in frame.intents}
    return QuerySignals(
        query=query,
        query_lower=query_lower,
        query_tokens=tokenize(query),
        query_frame=frame,
        normalized_emails=normalized_emails,
        current_state_hint=any(token in query_lower for token in {"current", "currently", "now", "active"}),
        relationship_hint="relation_lookup" in intent_values or bool(normalized_emails),
        temporal_hint="temporal_lookup" in intent_values,
        task_hint="task_lookup" in intent_values,
        identity_hint=bool(normalized_emails) or "entity_lookup" in intent_values,
    )


def search_records(
    records: Iterable[MemoryRecord],
    query: str,
    kinds,
    limit: int | None = None,
    score_boosts: dict[str, float] | None = None,
) -> list[tuple[MemoryRecord, float]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scored: list[tuple[MemoryRecord, float]] = []
    for record in records:
        if record.status is not RecordStatus.ACTIVE:
            continue
        if kinds and record.kind not in kinds:
            continue

        haystack = " ".join(
            [
                record.title or "",
                record.content,
                " ".join(record.tags),
            ]
        )
        score = score_tokens(query_tokens, tokenize(haystack))
        if score > 0:
            final_score = (
                score
                + score_tokens(query_tokens, tokenize(record.title or "")) * 0.15
                + (score_boosts or {}).get(record.memory_id, 0.0)
            )
            scored.append((record, final_score))

    scored.sort(key=lambda item: item[1], reverse=True)
    if limit is not None:
        return scored[:limit]
    return scored


def build_passive_response(
    matches: list[tuple[MemoryRecord, float]],
    *,
    query: str,
    max_results: int,
    token_budget: int,
    include_breakdown: bool = False,
    diagnostics: PassiveRecallDiagnostics | None = None,
) -> PassiveRecallResponse:
    rerank_started = perf_counter()
    items = [
        recall_item_from_record(record, score=score)
        for record, score in matches
    ]
    items = rerank_passive_items(items, query=query, include_breakdown=include_breakdown)
    rerank_ms = (perf_counter() - rerank_started) * 1000.0
    select_started = perf_counter()
    selected, total_token_count = select_passive_items(
        items,
        max_results=max_results,
        token_budget=token_budget,
    )
    select_ms = (perf_counter() - select_started) * 1000.0
    if diagnostics is not None:
        diagnostics.timings_ms["rerank_ms"] = round(rerank_ms, 3)
        diagnostics.timings_ms["selection_ms"] = round(select_ms, 3)
        diagnostics.counters["candidate_count"] = len(items)
        diagnostics.counters["selected_count"] = len(selected)
    return PassiveRecallResponse(
        items=selected,
        total_token_count=total_token_count,
        diagnostics=diagnostics,
    )


def build_active_response(
    matches: list[tuple[MemoryRecord, float]],
    *,
    diagnostics: ActiveRecallDiagnostics | None = None,
) -> ActiveRecallResponse:
    items = [
        RecallItem(
            memory_id=record.memory_id,
            kind=record.kind,
            title=record.title,
            content=record.content,
            score=score,
            tags=record.tags,
            token_count=approx_token_count(record.content),
        )
        for record, score in matches
    ]

    entities: dict[str, GraphEntity] = {}
    relations: list[GraphRelation] = []
    episodes: dict[str, GraphEpisodeItem] = {}
    for record, _ in matches:
        graph_entities, graph_relations = _graph_metadata_from_record(record)
        graph_episode = _graph_episode_from_record(record)
        for entity in graph_entities:
            entity_id = entity.get("entity_id") or entity.get("name")
            if not entity_id:
                continue

            existing = entities.get(entity_id)
            if existing is None:
                entities[entity_id] = GraphEntity(
                    entity_id=entity_id,
                    name=entity.get("name", entity_id),
                    entity_type=entity.get("entity_type", "unknown"),
                    memory_ids=[record.memory_id],
                )
            elif record.memory_id not in existing.memory_ids:
                existing.memory_ids.append(record.memory_id)

        for relation in graph_relations:
            relations.append(
                GraphRelation(
                    relation_id=relation.get(
                        "relation_id", f"rel_{record.memory_id}_{len(relations)}"
                    ),
                    source_entity_id=relation["source_entity_id"],
                    target_entity_id=relation["target_entity_id"],
                    relation_type=relation["relation_type"],
                    fact=relation["fact"],
                    memory_ids=[record.memory_id],
                    episode_ids=list(relation.get("episode_ids") or []),
                    valid_at=relation.get("valid_at"),
                    invalid_at=relation.get("invalid_at"),
                    invalidated_by_episode_id=relation.get("invalidated_by_episode_id"),
                )
            )
        if graph_episode is not None:
            episodes[graph_episode.episode_id] = graph_episode

    return ActiveRecallResponse(
        items=items,
        entities=list(entities.values()),
        relations=relations,
        episodes=list(episodes.values()),
        search_plan=[
            "lexical score over active canonical records",
            "expand metadata-provided entities, relations, and episode provenance",
        ],
        diagnostics=diagnostics,
    )


def _graph_metadata_from_record(record: MemoryRecord) -> tuple[list[dict], list[dict]]:
    graph_document = record.metadata.get("graph_document")
    if graph_document:
        entities_payload = graph_document.get("entities", []) or []
        relations_payload = graph_document.get("relations", []) or []
        entity_id_map: dict[str, str] = {}
        normalized_entities: list[dict] = []
        for entity in entities_payload:
            local_ref = entity.get("local_ref") or entity.get("canonical_name") or "entity"
            entity_id = f"{record.memory_id}:{local_ref}"
            entity_id_map[local_ref] = entity_id
            normalized_entities.append(
                {
                    "entity_id": entity_id,
                    "name": entity.get("canonical_name", entity_id),
                    "entity_type": entity.get("entity_type", "unknown"),
                }
            )
        normalized_relations: list[dict] = []
        for relation in relations_payload:
            source_ref = relation.get("source_ref")
            target_ref = relation.get("target_ref")
            if not source_ref or not target_ref:
                continue
            normalized_relations.append(
                {
                    "relation_id": f"rel_{record.memory_id}_{len(normalized_relations)}",
                    "source_entity_id": entity_id_map.get(source_ref, f"{record.memory_id}:{source_ref}"),
                    "target_entity_id": entity_id_map.get(target_ref, f"{record.memory_id}:{target_ref}"),
                    "relation_type": relation.get("relation_type", "mentions"),
                    "fact": relation.get("fact", ""),
                    "episode_ids": [episode["episode_id"]] if (episode := graph_document.get("episode")) else [],
                    "valid_at": relation.get("valid_at"),
                    "invalid_at": relation.get("invalid_at"),
                    "invalidated_by_episode_id": relation.get("invalidated_by_episode_id"),
                }
            )
        return normalized_entities, normalized_relations
    return record.metadata.get("entities", []), record.metadata.get("relations", [])


def _graph_episode_from_record(record: MemoryRecord) -> GraphEpisodeItem | None:
    graph_document = record.metadata.get("graph_document")
    if not isinstance(graph_document, dict):
        return None
    episode = graph_document.get("episode")
    if not isinstance(episode, dict):
        return None
    return GraphEpisodeItem.model_validate(episode)


def build_active_response_with_graph(
    *,
    matches: list[tuple[MemoryRecord, float]],
    graph_result: GraphSearchResult,
    diagnostics: ActiveRecallDiagnostics | None = None,
) -> ActiveRecallResponse:
    response = build_active_response(matches, diagnostics=diagnostics)
    response.entities = [
        GraphEntity(
            entity_id=entity.entity_id,
            name=entity.canonical_name,
            entity_type=entity.entity_type.value,
            memory_ids=entity.memory_ids,
        )
        for entity in graph_result.entities
    ]
    response.relations = [
        GraphRelation(
            relation_id=relation.relation_id,
            source_entity_id=relation.source_entity_id,
            target_entity_id=relation.target_entity_id,
            relation_type=relation.relation_type.value,
            fact=relation.fact,
            memory_ids=relation.memory_ids,
            episode_ids=relation.episode_ids,
            valid_at=relation.valid_at,
            invalid_at=relation.invalid_at,
            invalidated_by_episode_id=relation.invalidated_by_episode_id,
        )
        for relation in graph_result.relations
    ]
    response.episodes = [
        GraphEpisodeItem(
            episode_id=episode.episode_id,
            memory_id=episode.memory_id,
            source_type=episode.source_type,
            provenance_source_kind=episode.provenance_source_kind,
            provenance_source_id=episode.provenance_source_id,
            session_id=episode.session_id,
            task_id=episode.task_id,
            channel=episode.channel,
            created_at=episode.created_at,
            extracted_at=episode.extracted_at,
            extraction_confidence=episode.extraction_confidence,
            rationale=episode.rationale,
            source_excerpt=episode.source_excerpt,
            produced_relation_ids=episode.produced_relation_ids,
            invalidated_relation_ids=episode.invalidated_relation_ids,
        )
        for episode in graph_result.episodes
    ]
    response.search_plan = list(graph_result.search_plan)
    if graph_result.supporting_memory_ids:
        response.search_plan.append("load supporting canonical memories from graph hits")
    return response


def merge_passive_with_graph(
    *,
    base_response: PassiveRecallResponse,
    graph_records: list[MemoryRecord],
    query: str,
    kinds,
    max_results: int,
    token_budget: int,
    graph_bonus: float = 0.35,
    graph_memory_boosts: dict[str, float] | None = None,
    include_breakdown: bool = False,
) -> PassiveRecallResponse:
    item_map: dict[str, RecallItem] = {item.memory_id: item for item in base_response.items}
    signals = build_query_signals(query)
    graph_memory_boosts = graph_memory_boosts or {}

    for record in graph_records:
        if record.status is not RecordStatus.ACTIVE:
            continue
        if kinds and record.kind not in kinds:
            continue

        score = score_tokens(
            signals.query_tokens,
            tokenize(" ".join([record.title or "", record.content, " ".join(record.tags)])),
        )
        score += score_tokens(signals.query_tokens, tokenize(record.title or "")) * 0.15
        score += graph_memory_boosts.get(record.memory_id, graph_bonus)

        existing = item_map.get(record.memory_id)
        if existing is not None:
            existing.score = max(existing.score, score)
            existing.token_count = existing.token_count or approx_token_count(record.content)
            continue

        item_map[record.memory_id] = RecallItem(
            **recall_item_from_record(record, score=score).model_dump()
        )

    rerank_started = perf_counter()
    ranked_items = rerank_passive_items(
        list(item_map.values()),
        query=query,
        graph_memory_boosts={
            record.memory_id: graph_memory_boosts.get(record.memory_id, graph_bonus)
            for record in graph_records
        },
        include_breakdown=include_breakdown,
    )
    rerank_ms = (perf_counter() - rerank_started) * 1000.0
    select_started = perf_counter()
    selected, total_token_count = select_passive_items(
        ranked_items,
        max_results=max_results,
        token_budget=token_budget,
    )
    select_ms = (perf_counter() - select_started) * 1000.0
    diagnostics = base_response.diagnostics
    if diagnostics is not None:
        diagnostics.counters["graph_candidate_count"] = len(graph_records)
        diagnostics.counters["candidate_count"] = len(item_map)
        diagnostics.counters["selected_count"] = len(selected)
        diagnostics.flags["graph_assist_used"] = bool(graph_records)
        diagnostics.timings_ms["graph_merge_rerank_ms"] = round(rerank_ms, 3)
        diagnostics.timings_ms["graph_merge_selection_ms"] = round(select_ms, 3)
    return PassiveRecallResponse(
        items=selected,
        total_token_count=total_token_count,
        diagnostics=diagnostics,
    )


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9_]+", text.lower()) if token}


def score_tokens(query_tokens: set[str], content_tokens: set[str]) -> float:
    overlap = len(query_tokens & content_tokens)
    if overlap == 0:
        return 0.0
    return overlap / len(query_tokens)


def limit_recall_items(
    items: list[RecallItem],
    *,
    max_results: int,
    token_budget: int,
) -> tuple[list[RecallItem], int]:
    return select_passive_items(items, max_results=max_results, token_budget=token_budget)


def passive_candidate_limit(
    max_results: int,
    *,
    multiplier: int = 4,
    floor: int = 12,
    cap: int = 40,
) -> int:
    return min(max(max_results * multiplier, floor), cap)


def select_passive_items(
    items: list[RecallItem],
    *,
    max_results: int,
    token_budget: int,
) -> tuple[list[RecallItem], int]:
    selected: list[RecallItem] = []
    total_token_count = 0
    seen_ids: set[str] = set()
    if not items:
        return selected, total_token_count

    ranked_by_score = sorted(items, key=lambda item: item.score, reverse=True)
    if token_budget > 0:
        anchor_candidates = [
            item
            for item in ranked_by_score
            if (item.token_count or approx_token_count(item.content)) <= token_budget
        ]
        if not anchor_candidates:
            return selected, total_token_count
        top_anchor = max(anchor_candidates, key=_passive_utility_score)
    else:
        top_anchor = ranked_by_score[0]

    anchor_tokens = top_anchor.token_count or approx_token_count(top_anchor.content)
    selected.append(top_anchor)
    seen_ids.add(top_anchor.memory_id)
    total_token_count += anchor_tokens

    remaining = [item for item in ranked_by_score if item.memory_id not in seen_ids]
    remaining.sort(
        key=lambda item: _passive_utility_score(item),
        reverse=True,
    )

    duplicate_kind_counts: dict[MemoryKind, int] = {top_anchor.kind: 1}
    for item in remaining:
        if len(selected) >= max_results:
            break

        token_count = item.token_count or approx_token_count(item.content)
        if token_budget > 0 and selected and total_token_count + token_count > token_budget:
            continue

        kind_count = duplicate_kind_counts.get(item.kind, 0)
        utility_score = _passive_utility_score(item) - (kind_count * 0.03)
        if utility_score <= 0:
            continue

        selected.append(item)
        seen_ids.add(item.memory_id)
        total_token_count += token_count
        duplicate_kind_counts[item.kind] = kind_count + 1

        if token_budget > 0 and total_token_count >= token_budget:
            break

    return selected, total_token_count


def rerank_passive_items(
    items: list[RecallItem],
    *,
    query: str,
    graph_memory_boosts: dict[str, float] | None = None,
    include_breakdown: bool = False,
) -> list[RecallItem]:
    signals = build_query_signals(query)
    graph_memory_boosts = graph_memory_boosts or {}

    reranked: list[RecallItem] = []
    for item in items:
        adjusted = item.model_copy(deep=True)
        base_score = adjusted.score
        passive_bonus, breakdown = _passive_bonus(
            adjusted,
            signals=signals,
            graph_memory_boosts=graph_memory_boosts,
        )
        adjusted.score = adjusted.score + passive_bonus
        if include_breakdown:
            adjusted.score_breakdown = {"base": round(base_score, 6)}
            adjusted.score_breakdown.update(
                {
                    key: round(value, 6)
                    for key, value in breakdown.items()
                    if abs(value) > 0
                }
            )
            adjusted.score_breakdown["final"] = round(adjusted.score, 6)
        reranked.append(adjusted)

    reranked.sort(key=lambda item: item.score, reverse=True)
    return reranked


def kind_priority_bias(kind: MemoryKind) -> float:
    priorities = {
        MemoryKind.CORE_FACT: 0.20,
        MemoryKind.AGENT_NOTE: 0.15,
        MemoryKind.SESSION_SUMMARY: 0.05,
        MemoryKind.TASK_SUMMARY: 0.05,
        MemoryKind.USER_DATA: 0.00,
        MemoryKind.TRANSCRIPT: -0.10,
    }
    return priorities.get(kind, 0.0)


def recency_bias(updated_at: datetime | str | None) -> float:
    if not updated_at:
        return 0.0

    if isinstance(updated_at, str):
        try:
            timestamp = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
    else:
        timestamp = updated_at

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    age_days = max((datetime.now(timezone.utc) - timestamp).total_seconds() / 86_400, 0.0)
    return (1.0 / (1.0 + age_days / 30.0)) * 0.10


def approx_token_count(text: str) -> int:
    return max(len(re.findall(r"[A-Za-z0-9_]+", text)), 1)


def recall_item_from_record(record: MemoryRecord, *, score: float) -> RecallItem:
    confidence = _coerce_confidence(record.metadata.get("confidence"))
    return RecallItem(
        memory_id=record.memory_id,
        kind=record.kind,
        title=record.title,
        content=record.content,
        score=score,
        tags=record.tags,
        token_count=approx_token_count(record.content),
        source_kind=record.provenance.source_kind if record.provenance else None,
        updated_at=record.updated_at,
        confidence=confidence,
        canonical_key=_coerce_str(record.metadata.get("canonical_key")),
        always_include=_coerce_bool(record.metadata.get("always_include")),
        supersedes=record.supersedes,
    )


def recall_item_from_payload(payload: dict, *, score: float, point_id=None) -> RecallItem:
    raw_kind = payload.get("type")
    if raw_kind is None:
        raise KeyError(f"Missing `type` in recall payload for point {point_id}")
    kind = MemoryKind(raw_kind)
    return RecallItem(
        memory_id=str(payload.get("memory_id") or point_id),
        kind=kind,
        title=payload.get("title"),
        content=payload.get("content", ""),
        score=score,
        tags=list(payload.get("tags", [])),
        token_count=int(payload.get("token_count") or approx_token_count(payload.get("content", ""))),
        source_kind=_coerce_str(payload.get("source_kind")),
        updated_at=payload.get("updated_at"),
        confidence=_coerce_confidence(payload.get("confidence")),
        canonical_key=_coerce_str(payload.get("canonical_key")),
        always_include=_coerce_bool(payload.get("always_include")),
        supersedes=_coerce_str(payload.get("supersedes")),
    )


def _passive_bonus(
    item: RecallItem,
    *,
    signals: QuerySignals,
    graph_memory_boosts: dict[str, float],
) -> tuple[float, dict[str, float]]:
    bonus = 0.0
    breakdown: dict[str, float] = {}
    text_lower = " ".join([item.title or "", item.content, " ".join(item.tags)]).lower()
    if item.memory_id in graph_memory_boosts:
        bonus += _record_breakdown(
            breakdown,
            "graph_support",
            graph_memory_boosts[item.memory_id],
        )
    if signals.query_lower and len(signals.query_lower) >= 12 and signals.query_lower in text_lower:
        bonus += _record_breakdown(breakdown, "phrase_match", 0.10)
    if signals.current_state_hint and {"active", "current", "ongoing"} & {tag.lower() for tag in item.tags}:
        bonus += _record_breakdown(breakdown, "current_state_match", 0.08)
    if item.title:
        bonus += _record_breakdown(
            breakdown,
            "title_overlap",
            score_tokens(signals.query_tokens, tokenize(item.title)) * 0.12,
        )
    for email in signals.normalized_emails:
        if email in text_lower:
            bonus += _record_breakdown(breakdown, "exact_email", 0.70)
            break
    bonus += _record_breakdown(breakdown, "kind_bias", kind_priority_bias(item.kind))
    bonus += _record_breakdown(breakdown, "recency_bias", recency_bias(item.updated_at))
    bonus += _record_breakdown(breakdown, "intent_kind_bias", _intent_kind_bias(item.kind, signals))
    bonus += _record_breakdown(breakdown, "source_kind_bias", _source_kind_bias(item.source_kind))
    bonus += _record_breakdown(
        breakdown,
        "confidence_bias",
        _confidence_bias(item.confidence),
    )
    bonus += _record_breakdown(
        breakdown,
        "current_truth_bias",
        _current_truth_bias(item, signals),
    )
    token_count = item.token_count or approx_token_count(item.content)
    if token_count > 1400:
        bonus += _record_breakdown(breakdown, "token_size", -0.06)
    elif token_count < 220:
        bonus += _record_breakdown(breakdown, "token_size", 0.03)
    return bonus, breakdown


def _passive_utility_score(item: RecallItem) -> float:
    token_count = item.token_count or approx_token_count(item.content)
    return item.score / ((1.0 + (token_count / 256.0)) ** 0.5)


def _intent_kind_bias(kind: MemoryKind, signals: QuerySignals) -> float:
    bias = 0.0
    if signals.identity_hint:
        bias += {
            MemoryKind.AGENT_NOTE: 0.06,
            MemoryKind.USER_DATA: 0.04,
            MemoryKind.CORE_FACT: 0.02,
        }.get(kind, 0.0)
    if signals.relationship_hint:
        bias += {
            MemoryKind.AGENT_NOTE: 0.05,
            MemoryKind.SESSION_SUMMARY: 0.02,
            MemoryKind.TASK_SUMMARY: 0.03,
        }.get(kind, 0.0)
    if signals.temporal_hint:
        bias += {
            MemoryKind.SESSION_SUMMARY: 0.06,
            MemoryKind.TASK_SUMMARY: 0.04,
            MemoryKind.CORE_FACT: 0.02,
        }.get(kind, 0.0)
    if signals.task_hint:
        bias += {
            MemoryKind.TASK_SUMMARY: 0.08,
            MemoryKind.AGENT_NOTE: 0.04,
            MemoryKind.SESSION_SUMMARY: 0.03,
        }.get(kind, 0.0)
    return min(bias, 0.16)


def _source_kind_bias(source_kind: str | None) -> float:
    if not source_kind:
        return 0.0
    normalized = source_kind.strip().lower()
    return {
        "gateway": 0.06,
        "session_manager": 0.06,
        "orchestrator": 0.05,
        "agent": 0.03,
        "benchmark": 0.0,
        "test": 0.0,
    }.get(normalized, 0.0)


def _confidence_bias(confidence: float | None) -> float:
    if confidence is None:
        return 0.0
    bounded = max(0.0, min(confidence, 1.0))
    return (bounded - 0.5) * 0.12


def _current_truth_bias(item: RecallItem, signals: QuerySignals) -> float:
    bias = 0.0
    if item.kind is MemoryKind.CORE_FACT:
        bias += 0.06
    if item.always_include:
        bias += 0.04
    if item.canonical_key:
        bias += 0.03
    if item.supersedes:
        bias += 0.04
    if not signals.current_state_hint and {"active", "current", "ongoing"} & {
        tag.lower() for tag in item.tags
    }:
        bias += 0.02
    return bias


def _record_breakdown(breakdown: dict[str, float], key: str, value: float) -> float:
    if value == 0:
        return 0.0
    breakdown[key] = breakdown.get(key, 0.0) + value
    return value


def _coerce_confidence(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def apply_graph_recipe_for_mode(
    *,
    graph_result: GraphSearchResult,
    query_frame: GraphQueryFrame,
    mode: str,
    max_results: int,
) -> GraphRecipeApplication:
    return apply_graph_search_recipe(
        graph_result=graph_result,
        query_frame=query_frame,
        mode=mode,
        max_results=max_results,
    )
