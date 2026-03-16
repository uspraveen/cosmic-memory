"""Benchmark real-data retrieval latency across graph backends."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cosmic_memory.domain.models import ActiveRecallRequest, GraphSyncRequest, PassiveRecallRequest
from cosmic_memory.env import load_env_file
from cosmic_memory.filesystem_service import FilesystemMemoryService
from cosmic_memory.graph import InMemoryGraphStore, Neo4jGraphStore
from cosmic_memory.server.app import _build_embedding_service_from_env, _build_passive_index_from_env


PASSIVE_QUERIES = (
    "yc",
    "What does Praveen prefer?",
    "DeepAgents",
)

ACTIVE_QUERIES = (
    "What does Praveen prefer?",
    "What happened after the YC S26 rejection?",
    "What did we decide about DeepAgents?",
)


@dataclass(slots=True)
class Sample:
    mode: str
    operation: str
    query: str
    latency_ms: float
    result_count: int
    graph_used: bool
    graph_timed_out: bool
    timings_ms: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        default="none,memory,neo4j",
        help="Comma-separated list from: none,memory,neo4j",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--passive-max-results", type=int, default=6)
    parser.add_argument("--passive-token-budget", type=int, default=1200)
    parser.add_argument("--active-max-results", type=int, default=8)
    parser.add_argument("--active-max-hops", type=int, default=2)
    parser.add_argument("--graph-timeout-ms", type=int, default=120)
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


async def benchmark_mode(
    *,
    mode: str,
    data_dir: str,
    iterations: int,
    warmup: int,
    passive_max_results: int,
    passive_token_budget: int,
    active_max_results: int,
    active_max_hops: int,
    graph_timeout_ms: int,
) -> dict:
    embedding_service = _build_embedding_service_from_env(require_remote=True)
    passive_index = _build_passive_index_from_env(
        embedding_service,
        default_path=str(Path(data_dir) / "qdrant_data"),
    )
    graph_store = build_graph_store(mode=mode)
    service = FilesystemMemoryService(
        data_dir,
        passive_index=passive_index,
        graph_store=graph_store,
        passive_graph_timeout_seconds=graph_timeout_ms / 1000.0,
    )
    try:
        if mode == "memory":
            await service.sync_graph(
                GraphSyncRequest(
                    allow_llm=False,
                    persist_graph_documents=False,
                    only_missing_graph_documents=False,
                    warm_cache=False,
                )
            )
        elif mode == "neo4j":
            await service.warm_graph_cache()

        status = await service.get_graph_status()
        samples = await run_samples(
            service=service,
            mode=mode,
            iterations=iterations,
            warmup=warmup,
            passive_max_results=passive_max_results,
            passive_token_budget=passive_token_budget,
            active_max_results=active_max_results,
            active_max_hops=active_max_hops,
        )
        return {
            "mode": mode,
            "status": status.model_dump(mode="json"),
            "passive": summarize(samples, operation="passive"),
            "active": summarize(samples, operation="active"),
            "samples": [asdict(sample) for sample in samples],
        }
    finally:
        await close_if_present(graph_store)
        await close_if_present(passive_index)
        await close_if_present(embedding_service)


async def run_samples(
    *,
    service: FilesystemMemoryService,
    mode: str,
    iterations: int,
    warmup: int,
    passive_max_results: int,
    passive_token_budget: int,
    active_max_results: int,
    active_max_hops: int,
) -> list[Sample]:
    for _ in range(max(warmup, 0)):
        for query in PASSIVE_QUERIES:
            await invoke_passive(
                service,
                mode=mode,
                query=query,
                max_results=passive_max_results,
                token_budget=passive_token_budget,
            )
        for query in ACTIVE_QUERIES:
            await invoke_active(
                service,
                mode=mode,
                query=query,
                max_results=active_max_results,
                max_hops=active_max_hops,
            )

    samples: list[Sample] = []
    for _ in range(max(iterations, 1)):
        for query in PASSIVE_QUERIES:
            samples.append(
                await invoke_passive(
                    service,
                    mode=mode,
                    query=query,
                    max_results=passive_max_results,
                    token_budget=passive_token_budget,
                )
            )
        for query in ACTIVE_QUERIES:
            samples.append(
                await invoke_active(
                    service,
                    mode=mode,
                    query=query,
                    max_results=active_max_results,
                    max_hops=active_max_hops,
                )
            )
    return samples


async def invoke_passive(
    service: FilesystemMemoryService,
    *,
    mode: str,
    query: str,
    max_results: int,
    token_budget: int,
) -> Sample:
    started = perf_counter()
    response = await service.passive_recall(
        PassiveRecallRequest(
            query=query,
            max_results=max_results,
            token_budget=token_budget,
            include_diagnostics=True,
        )
    )
    latency_ms = round((perf_counter() - started) * 1000.0, 3)
    diagnostics = response.diagnostics.model_dump(mode="json") if response.diagnostics else {}
    flags = diagnostics.get("flags") or {}
    timings = diagnostics.get("timings_ms") or {}
    return Sample(
        mode=mode,
        operation="passive",
        query=query,
        latency_ms=latency_ms,
        result_count=len(response.items),
        graph_used=bool(flags.get("graph_assist_used")),
        graph_timed_out=bool(flags.get("graph_assist_timed_out")),
        timings_ms={key: float(value) for key, value in timings.items()},
    )


async def invoke_active(
    service: FilesystemMemoryService,
    *,
    mode: str,
    query: str,
    max_results: int,
    max_hops: int,
) -> Sample:
    started = perf_counter()
    response = await service.active_recall(
        ActiveRecallRequest(
            query=query,
            max_results=max_results,
            max_hops=max_hops,
            include_diagnostics=True,
        )
    )
    latency_ms = round((perf_counter() - started) * 1000.0, 3)
    diagnostics = response.diagnostics.model_dump(mode="json") if response.diagnostics else {}
    flags = diagnostics.get("flags") or {}
    timings = diagnostics.get("timings_ms") or {}
    return Sample(
        mode=mode,
        operation="active",
        query=query,
        latency_ms=latency_ms,
        result_count=len(response.items),
        graph_used=bool(flags.get("graph_used")),
        graph_timed_out=False,
        timings_ms={key: float(value) for key, value in timings.items()},
    )


def summarize(samples: list[Sample], *, operation: str) -> dict:
    relevant = [sample for sample in samples if sample.operation == operation]
    latencies = [sample.latency_ms for sample in relevant]
    return {
        "samples": len(relevant),
        "mean_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "p50_ms": round(percentile(latencies, 0.50), 3) if latencies else 0.0,
        "p95_ms": round(percentile(latencies, 0.95), 3) if latencies else 0.0,
        "max_ms": round(max(latencies), 3) if latencies else 0.0,
        "graph_used_rate": round(
            sum(1 for sample in relevant if sample.graph_used) / max(len(relevant), 1),
            4,
        ),
        "graph_timeout_rate": round(
            sum(1 for sample in relevant if sample.graph_timed_out) / max(len(relevant), 1),
            4,
        ),
        "queries": {
            query: {
                "mean_ms": round(
                    statistics.fmean(sample.latency_ms for sample in relevant if sample.query == query),
                    3,
                ),
                "p95_ms": round(
                    percentile(
                        [sample.latency_ms for sample in relevant if sample.query == query],
                        0.95,
                    ),
                    3,
                ),
            }
            for query in sorted({sample.query for sample in relevant})
        },
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = max(min(int(round((len(ordered) - 1) * pct)), len(ordered) - 1), 0)
    return ordered[index]


def build_graph_store(*, mode: str):
    if mode == "none":
        return None
    if mode == "memory":
        return InMemoryGraphStore()
    return Neo4jGraphStore(
        uri=require_env("COSMIC_MEMORY_NEO4J_URI"),
        username=require_env("COSMIC_MEMORY_NEO4J_USERNAME"),
        password=require_env("COSMIC_MEMORY_NEO4J_PASSWORD"),
        database=os.environ.get("COSMIC_MEMORY_NEO4J_DATABASE", "neo4j"),
    )


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for neo4j benchmark mode.")
    return value


async def close_if_present(resource) -> None:
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if close is None:
        return
    await close()


async def async_main() -> None:
    args = parse_args()
    load_env_file()
    data_dir = args.data_dir or os.environ.get("COSMIC_MEMORY_DATA_DIR")
    if not data_dir:
        raise SystemExit("--data-dir or COSMIC_MEMORY_DATA_DIR is required.")
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    report = {
        "data_dir": data_dir,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "modes": [],
    }
    for mode in modes:
        report["modes"].append(
            await benchmark_mode(
                mode=mode,
                data_dir=data_dir,
                iterations=args.iterations,
                warmup=args.warmup,
                passive_max_results=args.passive_max_results,
                passive_token_budget=args.passive_token_budget,
                active_max_results=args.active_max_results,
                active_max_hops=args.active_max_hops,
                graph_timeout_ms=args.graph_timeout_ms,
            )
        )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
