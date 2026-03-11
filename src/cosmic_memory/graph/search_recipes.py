"""Internal graph retrieval recipes for passive and active memory search."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import re

from cosmic_memory.graph.models import (
    GraphEntityNode,
    GraphEpisode,
    GraphQueryFrame,
    GraphRelationEdge,
    GraphSearchResult,
)
from cosmic_memory.graph.ontology import QueryIntent


@dataclass(frozen=True, slots=True)
class GraphSearchRecipe:
    name: str
    mode: str
    use_rrf: bool = True
    use_mmr: bool = False
    use_node_distance: bool = True
    use_episode_mentions: bool = True
    mmr_lambda: float = 0.72
    supporting_memory_limit: int = 8
    rrf_k: int = 8
    rrf_weight: float = 2.5


@dataclass(frozen=True, slots=True)
class GraphRecipeApplication:
    recipe_name: str
    graph_result: GraphSearchResult
    memory_boosts: dict[str, float]


def apply_graph_search_recipe(
    *,
    graph_result: GraphSearchResult,
    query_frame: GraphQueryFrame,
    mode: str,
    max_results: int,
) -> GraphRecipeApplication:
    recipe = choose_graph_search_recipe(query_frame, mode=mode, max_results=max_results)
    if not graph_result.relations:
        updated_result = graph_result.model_copy(
            update={
                "search_plan": [
                    *graph_result.search_plan,
                    f"apply graph recipe: {recipe.name} (no relations to rerank)",
                ]
            }
        )
        return GraphRecipeApplication(
            recipe_name=recipe.name,
            graph_result=updated_result,
            memory_boosts={},
        )

    relation_scores = _score_relations(
        graph_result=graph_result,
        query_frame=query_frame,
        recipe=recipe,
    )
    ranked_relations = sorted(
        graph_result.relations,
        key=lambda relation: relation_scores.get(relation.relation_id, 0.0),
        reverse=True,
    )
    if recipe.use_mmr:
        ranked_relations = _select_relations_with_mmr(
            ranked_relations,
            relation_scores=relation_scores,
            query_frame=query_frame,
            graph_result=graph_result,
            limit=max_results,
            lambda_value=recipe.mmr_lambda,
        )
    else:
        ranked_relations = ranked_relations[:max_results]

    selected_entities = _select_entities(
        graph_result=graph_result,
        relations=ranked_relations,
        entity_limit=max_results,
    )
    selected_episodes = _select_episodes(
        graph_result=graph_result,
        relations=ranked_relations,
        episode_limit=max_results,
    )
    memory_boosts = _memory_boosts_from_relations(
        relations=ranked_relations,
        episodes=selected_episodes,
        relation_scores=relation_scores,
    )
    supporting_memory_ids = [
        memory_id
        for memory_id, _score in sorted(
            memory_boosts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[: recipe.supporting_memory_limit]
    ]

    updated_result = graph_result.model_copy(
        update={
            "entities": selected_entities,
            "relations": ranked_relations,
            "episodes": selected_episodes,
            "supporting_memory_ids": supporting_memory_ids,
            "search_plan": [
                *graph_result.search_plan,
                f"apply graph recipe: {recipe.name}",
            ],
        }
    )
    return GraphRecipeApplication(
        recipe_name=recipe.name,
        graph_result=updated_result,
        memory_boosts=memory_boosts,
    )


def choose_graph_search_recipe(
    query_frame: GraphQueryFrame,
    *,
    mode: str,
    max_results: int,
) -> GraphSearchRecipe:
    supporting_limit = min(max(max_results * 2, 6), 16)
    intent_values = set(query_frame.intents)

    if mode == "passive":
        return GraphSearchRecipe(
            name="passive_hybrid_rrf",
            mode=mode,
            use_rrf=True,
            use_mmr=False,
            use_node_distance=True,
            use_episode_mentions=True,
            supporting_memory_limit=supporting_limit,
        )

    if QueryIntent.TEMPORAL_LOOKUP in intent_values:
        return GraphSearchRecipe(
            name="active_temporal_rrf_mmr",
            mode=mode,
            use_rrf=True,
            use_mmr=True,
            use_node_distance=True,
            use_episode_mentions=True,
            mmr_lambda=0.64,
            supporting_memory_limit=supporting_limit,
        )
    if query_frame.prefer_current_state:
        return GraphSearchRecipe(
            name="active_current_state_rrf_mmr",
            mode=mode,
            use_rrf=True,
            use_mmr=True,
            use_node_distance=True,
            use_episode_mentions=True,
            mmr_lambda=0.7,
            supporting_memory_limit=supporting_limit,
        )
    return GraphSearchRecipe(
        name="active_hybrid_rrf_mmr",
        mode=mode,
        use_rrf=True,
        use_mmr=True,
        use_node_distance=True,
        use_episode_mentions=True,
        mmr_lambda=0.55,
        supporting_memory_limit=supporting_limit,
    )


def _score_relations(
    *,
    graph_result: GraphSearchResult,
    query_frame: GraphQueryFrame,
    recipe: GraphSearchRecipe,
) -> dict[str, float]:
    entity_by_id = {entity.entity_id: entity for entity in graph_result.entities}
    episode_by_id = {episode.episode_id: episode for episode in graph_result.episodes}
    query_tokens = _tokenize(query_frame.query)

    lexical_scores: dict[str, float] = {}
    distance_scores: dict[str, float] = {}
    episode_scores: dict[str, float] = {}
    state_scores: dict[str, float] = {}

    for relation in graph_result.relations:
        lexical_scores[relation.relation_id] = _lexical_relation_score(
            relation=relation,
            entity_by_id=entity_by_id,
            query_tokens=query_tokens,
        )
        distance_scores[relation.relation_id] = _node_distance_score(
            relation=relation,
            graph_result=graph_result,
        )
        episode_scores[relation.relation_id] = _episode_mentions_score(
            relation=relation,
            episode_by_id=episode_by_id,
        )
        state_scores[relation.relation_id] = _state_alignment_score(
            relation=relation,
            query_frame=query_frame,
        )

    if not recipe.use_rrf:
        return {
            relation.relation_id: lexical_scores.get(relation.relation_id, 0.0)
            + state_scores.get(relation.relation_id, 0.0)
            for relation in graph_result.relations
        }

    ranking_lists: list[list[str]] = [
        _rank_ids(lexical_scores),
        _rank_ids(state_scores),
    ]
    if recipe.use_node_distance:
        ranking_lists.append(_rank_ids(distance_scores))
    if recipe.use_episode_mentions:
        ranking_lists.append(_rank_ids(episode_scores))

    fused_scores = _rrf_fuse(ranking_lists, k=recipe.rrf_k)
    final_scores: dict[str, float] = {}
    for relation in graph_result.relations:
        relation_id = relation.relation_id
        final_scores[relation_id] = (
            fused_scores.get(relation_id, 0.0) * recipe.rrf_weight
            + lexical_scores.get(relation_id, 0.0) * 0.35
            + state_scores.get(relation_id, 0.0) * 0.35
            + distance_scores.get(relation_id, 0.0) * 0.20
            + episode_scores.get(relation_id, 0.0) * 0.20
        )
    return final_scores


def _select_relations_with_mmr(
    relations: list[GraphRelationEdge],
    *,
    relation_scores: dict[str, float],
    query_frame: GraphQueryFrame,
    graph_result: GraphSearchResult,
    limit: int,
    lambda_value: float,
) -> list[GraphRelationEdge]:
    entity_by_id = {entity.entity_id: entity for entity in graph_result.entities}
    remaining = list(relations)
    selected: list[GraphRelationEdge] = []

    while remaining and len(selected) < limit:
        if not selected:
            next_relation = max(
                remaining,
                key=lambda relation: relation_scores.get(relation.relation_id, 0.0),
            )
            selected.append(next_relation)
            remaining = [
                relation for relation in remaining if relation.relation_id != next_relation.relation_id
            ]
            continue

        best_relation: GraphRelationEdge | None = None
        best_score = -math.inf
        for candidate in remaining:
            relevance = relation_scores.get(candidate.relation_id, 0.0)
            diversity_penalty = max(
                _relation_similarity(candidate, chosen, entity_by_id=entity_by_id)
                for chosen in selected
            )
            mmr_score = (lambda_value * relevance) - ((1.0 - lambda_value) * diversity_penalty)
            if query_frame.prefer_current_state and _is_active_relation(candidate):
                mmr_score += 0.03
            if mmr_score > best_score:
                best_score = mmr_score
                best_relation = candidate

        if best_relation is None:
            break
        selected.append(best_relation)
        remaining = [
            relation for relation in remaining if relation.relation_id != best_relation.relation_id
        ]

    return selected


def _select_entities(
    *,
    graph_result: GraphSearchResult,
    relations: list[GraphRelationEdge],
    entity_limit: int,
) -> list[GraphEntityNode]:
    entity_by_id = {entity.entity_id: entity for entity in graph_result.entities}
    support_scores: dict[str, float] = {entity_id: 0.0 for entity_id in entity_by_id}

    for relation in relations:
        support_scores[relation.source_entity_id] = support_scores.get(relation.source_entity_id, 0.0) + 1.0
        support_scores[relation.target_entity_id] = support_scores.get(relation.target_entity_id, 0.0) + 0.9
    for seed_entity_id in graph_result.seed_entity_ids:
        support_scores[seed_entity_id] = support_scores.get(seed_entity_id, 0.0) + 0.75

    ranked_entity_ids = sorted(
        support_scores,
        key=lambda entity_id: (
            support_scores.get(entity_id, 0.0),
            entity_id in set(graph_result.seed_entity_ids),
        ),
        reverse=True,
    )
    return [
        entity_by_id[entity_id]
        for entity_id in ranked_entity_ids
        if entity_id in entity_by_id
    ][:entity_limit]


def _select_episodes(
    *,
    graph_result: GraphSearchResult,
    relations: list[GraphRelationEdge],
    episode_limit: int,
) -> list[GraphEpisode]:
    episode_by_id = {episode.episode_id: episode for episode in graph_result.episodes}
    support_scores: dict[str, float] = {}

    for relation in relations:
        for episode_id in relation.episode_ids:
            support_scores[episode_id] = support_scores.get(episode_id, 0.0) + 1.0
        if relation.invalidated_by_episode_id:
            support_scores[relation.invalidated_by_episode_id] = (
                support_scores.get(relation.invalidated_by_episode_id, 0.0) + 0.85
            )

    ranked_episode_ids = sorted(
        support_scores,
        key=lambda episode_id: (
            support_scores.get(episode_id, 0.0),
            _episode_recency_score(episode_by_id.get(episode_id)),
            episode_by_id.get(episode_id).extraction_confidence if episode_by_id.get(episode_id) else 0.0,
        ),
        reverse=True,
    )
    return [
        episode_by_id[episode_id]
        for episode_id in ranked_episode_ids
        if episode_id in episode_by_id
    ][:episode_limit]


def _memory_boosts_from_relations(
    *,
    relations: list[GraphRelationEdge],
    episodes: list[GraphEpisode],
    relation_scores: dict[str, float],
) -> dict[str, float]:
    boosts: dict[str, float] = {}
    for rank, relation in enumerate(relations):
        relation_score = relation_scores.get(relation.relation_id, 0.0)
        rank_bonus = 0.30 / (rank + 1)
        for memory_id in relation.memory_ids:
            boosts[memory_id] = boosts.get(memory_id, 0.0) + (relation_score * 0.08) + rank_bonus
    for episode in episodes:
        boosts[episode.memory_id] = boosts.get(episode.memory_id, 0.0) + min(
            0.12 + (episode.extraction_confidence * 0.08),
            0.22,
        )
    return boosts


def _lexical_relation_score(
    *,
    relation: GraphRelationEdge,
    entity_by_id: dict[str, GraphEntityNode],
    query_tokens: set[str],
) -> float:
    if not query_tokens:
        return 0.0
    text = " ".join(
        [
            relation.fact,
            relation.relation_type.value,
            entity_by_id.get(relation.source_entity_id).canonical_name
            if entity_by_id.get(relation.source_entity_id)
            else relation.source_entity_id,
            entity_by_id.get(relation.target_entity_id).canonical_name
            if entity_by_id.get(relation.target_entity_id)
            else relation.target_entity_id,
        ]
    )
    relation_tokens = _tokenize(text)
    overlap = len(query_tokens & relation_tokens)
    return overlap / max(len(query_tokens), 1)


def _node_distance_score(
    *,
    relation: GraphRelationEdge,
    graph_result: GraphSearchResult,
) -> float:
    distance = graph_result.relation_distances.get(relation.relation_id)
    if distance is None or distance <= 0:
        return 0.0
    return 1.0 / float(distance)


def _episode_mentions_score(
    *,
    relation: GraphRelationEdge,
    episode_by_id: dict[str, GraphEpisode],
) -> float:
    if not relation.episode_ids and not relation.invalidated_by_episode_id:
        return 0.0

    episode_ids = list(relation.episode_ids)
    if relation.invalidated_by_episode_id:
        episode_ids.append(relation.invalidated_by_episode_id)

    score = 0.0
    for episode_id in episode_ids:
        episode = episode_by_id.get(episode_id)
        if episode is None:
            continue
        score += 0.18
        score += episode.extraction_confidence * 0.10
        score += _episode_recency_score(episode) * 0.12
    return score


def _state_alignment_score(
    *,
    relation: GraphRelationEdge,
    query_frame: GraphQueryFrame,
) -> float:
    score = 0.0
    active = _is_active_relation(relation)
    if query_frame.prefer_current_state:
        if active:
            score += 0.30
        else:
            score -= 0.10
    else:
        if relation.invalidated_by_episode_id or relation.invalid_at is not None:
            score += 0.10
        if active:
            score += 0.05
    return score


def _rrf_fuse(rankings: list[list[str]], *, k: int = 8) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for index, item_id in enumerate(ranking, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + (1.0 / (k + index))
    return fused


def _rank_ids(scores: dict[str, float]) -> list[str]:
    return [
        item_id
        for item_id, _score in sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if _score > 0
    ]


def _relation_similarity(
    left: GraphRelationEdge,
    right: GraphRelationEdge,
    *,
    entity_by_id: dict[str, GraphEntityNode],
) -> float:
    left_tokens = _relation_tokens(left, entity_by_id=entity_by_id)
    right_tokens = _relation_tokens(right, entity_by_id=entity_by_id)
    if not left_tokens or not right_tokens:
        return 0.0

    token_similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    type_bonus = 0.15 if left.relation_type == right.relation_type else 0.0
    entity_bonus = 0.0
    if left.source_entity_id == right.source_entity_id:
        entity_bonus += 0.12
    if left.target_entity_id == right.target_entity_id:
        entity_bonus += 0.12
    return min(token_similarity + type_bonus + entity_bonus, 1.0)


def _relation_tokens(
    relation: GraphRelationEdge,
    *,
    entity_by_id: dict[str, GraphEntityNode],
) -> set[str]:
    return _tokenize(
        " ".join(
            [
                relation.fact,
                relation.relation_type.value,
                entity_by_id.get(relation.source_entity_id).canonical_name
                if entity_by_id.get(relation.source_entity_id)
                else relation.source_entity_id,
                entity_by_id.get(relation.target_entity_id).canonical_name
                if entity_by_id.get(relation.target_entity_id)
                else relation.target_entity_id,
            ]
        )
    )


def _episode_recency_score(episode: GraphEpisode | None) -> float:
    if episode is None:
        return 0.0
    timestamp = episode.created_at
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    age_days = max((datetime.now(timezone.utc) - timestamp).total_seconds() / 86_400.0, 0.0)
    return 1.0 / (1.0 + (age_days / 30.0))


def _is_active_relation(relation: GraphRelationEdge) -> bool:
    now = datetime.now(timezone.utc)
    if relation.invalidated_by_episode_id is not None:
        return False
    if relation.invalid_at and relation.invalid_at <= now:
        return False
    if relation.expires_at and relation.expires_at <= now:
        return False
    return True


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Za-z0-9_]+", text.casefold()) if token}
