"""Benchmark graph-backed active recall latency and result quality."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
import random
import statistics
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cosmic_memory.domain.enums import MemoryKind
from cosmic_memory.domain.models import (
    ActiveRecallRequest,
    MemoryProvenance,
    WriteMemoryRequest,
)
from cosmic_memory.env import load_env_file
from cosmic_memory.filesystem_service import FilesystemMemoryService
from cosmic_memory.graph import InMemoryGraphStore, Neo4jGraphStore


@dataclass(slots=True)
class BenchmarkCase:
    name: str
    query: str
    expected_relation: str
    expected_terms: tuple[str, ...]
    max_results: int = 8
    max_hops: int = 2


@dataclass(slots=True)
class BenchmarkSample:
    case_name: str
    latency_ms: float
    top_relation_hit: bool
    any_relation_hit: bool
    result_count: int
    relation_count: int
    timings_ms: dict[str, float]
    counters: dict[str, int]
    flags: dict[str, bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["inprocess", "http"], default="inprocess")
    parser.add_argument("--base-url", default=None, help="Required for HTTP mode.")
    parser.add_argument("--records", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--graph-backend", choices=["memory", "neo4j"], default="memory")
    parser.add_argument("--data-dir", default=None, help="Optional benchmark data directory.")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="benchmark", created_by="active_recall_benchmark")


def benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            name="identity_relationship",
            query="What project does nxagarwal@ualr.edu work on?",
            expected_relation="works_on",
            expected_terms=("cosmic memory",),
        ),
        BenchmarkCase(
            name="project_blocker",
            query="What is blocking Cosmic Memory right now?",
            expected_relation="blocked_by",
            expected_terms=("embedding latency",),
        ),
        BenchmarkCase(
            name="project_decision",
            query="What did we decide about Qdrant BM25 for Cosmic Memory?",
            expected_relation="decided",
            expected_terms=("qdrant native bm25",),
        ),
    ]


def synthetic_requests(total_records: int) -> list[WriteMemoryRequest]:
    fixed = [
        WriteMemoryRequest(
            kind=MemoryKind.AGENT_NOTE,
            title="Project relationship",
            content="Nitin Agarwal works on Cosmic Memory.",
            provenance=provenance(),
            metadata={
                "graph_document": {
                    "memory_id": "ignored",
                    "entities": [
                        {
                            "local_ref": "person",
                            "entity_type": "person",
                            "canonical_name": "Nitin Agarwal",
                            "identity_candidates": [
                                {
                                    "key_type": "email",
                                    "raw_value": "nxagarwal@ualr.edu",
                                }
                            ],
                            "alias_values": ["Dr. Nitin"],
                        },
                        {
                            "local_ref": "project",
                            "entity_type": "project",
                            "canonical_name": "Cosmic Memory",
                            "alias_values": ["Cosmic"],
                        },
                    ],
                    "relations": [
                        {
                            "source_ref": "person",
                            "target_ref": "project",
                            "relation_type": "works_on",
                            "fact": "Nitin Agarwal works on Cosmic Memory.",
                        }
                    ],
                }
            },
        ),
        WriteMemoryRequest(
            kind=MemoryKind.TASK_SUMMARY,
            title="Current blocker",
            content="Cosmic Memory is blocked by embedding latency work.",
            tags=["current", "active", "blocking"],
            provenance=provenance(),
            metadata={
                "graph_document": {
                    "memory_id": "ignored",
                    "entities": [
                        {
                            "local_ref": "project",
                            "entity_type": "project",
                            "canonical_name": "Cosmic Memory",
                        },
                        {
                            "local_ref": "blocker",
                            "entity_type": "artifact",
                            "canonical_name": "Embedding latency",
                        },
                    ],
                    "relations": [
                        {
                            "source_ref": "project",
                            "target_ref": "blocker",
                            "relation_type": "blocked_by",
                            "fact": "Cosmic Memory is blocked by embedding latency.",
                        }
                    ],
                }
            },
        ),
        WriteMemoryRequest(
            kind=MemoryKind.SESSION_SUMMARY,
            title="Retrieval decision",
            content="We decided to keep Qdrant native BM25 for Cosmic Memory passive recall.",
            tags=["decision", "qdrant", "bm25"],
            provenance=provenance(),
            metadata={
                "graph_document": {
                    "memory_id": "ignored",
                    "entities": [
                        {
                            "local_ref": "project",
                            "entity_type": "project",
                            "canonical_name": "Cosmic Memory",
                        },
                        {
                            "local_ref": "decision",
                            "entity_type": "artifact",
                            "canonical_name": "Qdrant native BM25",
                        },
                    ],
                    "relations": [
                        {
                            "source_ref": "project",
                            "target_ref": "decision",
                            "relation_type": "decided",
                            "fact": "We decided to keep Qdrant native BM25 for Cosmic Memory passive recall.",
                        }
                    ],
                }
            },
        ),
    ]

    randomizer = random.Random(42)
    filler_topics = [
        "gardening",
        "podcasts",
        "astronomy",
        "emails",
        "finance",
        "recipes",
        "travel",
        "linux",
        "papers",
        "meeting notes",
    ]
    filler_count = max(total_records - len(fixed), 0)
    for index in range(filler_count):
        topic = filler_topics[index % len(filler_topics)]
        noise = " ".join(randomizer.sample(filler_topics, k=min(4, len(filler_topics))))
        fixed.append(
            WriteMemoryRequest(
                kind=MemoryKind.AGENT_NOTE,
                title=f"Synthetic graph note {index + 1}",
                content=f"Synthetic graph memory about {topic}. Related notes: {noise}.",
                tags=[topic, "synthetic"],
                provenance=provenance(),
            )
        )
    return fixed


async def seed_inprocess_service(service: FilesystemMemoryService, total_records: int) -> None:
    for request in synthetic_requests(total_records):
        await service.write(request)


async def seed_http_service(client: httpx.AsyncClient, total_records: int) -> None:
    for request in synthetic_requests(total_records):
        response = await client.post("/v1/memories", json=request.model_dump(mode="json"))
        response.raise_for_status()


async def run_inprocess_benchmark(args: argparse.Namespace) -> dict:
    load_env_file()
    graph_store = build_graph_store(args)
    try:
        with benchmark_workdir(args.data_dir) as tmp:
            service = FilesystemMemoryService(tmp, graph_store=graph_store)
            await seed_inprocess_service(service, args.records)
            return await _exercise_active_recall(
                lambda case: service.active_recall(
                    ActiveRecallRequest(
                        query=case.query,
                        max_results=case.max_results,
                        max_hops=case.max_hops,
                        include_diagnostics=True,
                    )
                ),
                warmup=args.warmup,
                iterations=args.iterations,
                concurrency=args.concurrency,
            )
    finally:
        await close_if_present(graph_store)


async def run_http_benchmark(args: argparse.Namespace) -> dict:
    if not args.base_url:
        raise SystemExit("--base-url is required when --mode=http")
    async with httpx.AsyncClient(base_url=args.base_url, timeout=30.0) as client:
        await seed_http_service(client, args.records)
        return await _exercise_active_recall(
            lambda case: client.post(
                "/v1/query/active",
                json=ActiveRecallRequest(
                    query=case.query,
                    max_results=case.max_results,
                    max_hops=case.max_hops,
                    include_diagnostics=True,
                ).model_dump(mode="json"),
            ),
            warmup=args.warmup,
            iterations=args.iterations,
            concurrency=args.concurrency,
            http_mode=True,
        )


def build_graph_store(args: argparse.Namespace):
    import os

    if args.graph_backend == "memory":
        return InMemoryGraphStore()

    uri = require_env("COSMIC_MEMORY_NEO4J_URI")
    username = require_env("COSMIC_MEMORY_NEO4J_USERNAME")
    password = require_env("COSMIC_MEMORY_NEO4J_PASSWORD")
    return Neo4jGraphStore(
        uri=uri,
        username=username,
        password=password,
        database=os.environ.get("COSMIC_MEMORY_NEO4J_DATABASE", "neo4j"),
    )


def require_env(name: str) -> str:
    import os

    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required for the Neo4j benchmark path.")
    return value


async def close_if_present(resource) -> None:
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if close is None:
        return
    await close()


async def _exercise_active_recall(
    caller,
    *,
    warmup: int,
    iterations: int,
    concurrency: int,
    http_mode: bool = False,
) -> dict:
    cases = benchmark_cases()
    for _ in range(max(warmup, 0)):
        for case in cases:
            await _invoke_case(caller, case, http_mode=http_mode)

    scheduled_cases = [cases[index % len(cases)] for index in range(iterations * len(cases))]
    samples: list[BenchmarkSample] = []
    for offset in range(0, len(scheduled_cases), max(concurrency, 1)):
        batch = scheduled_cases[offset : offset + max(concurrency, 1)]
        batch_samples = await asyncio.gather(
            *[_invoke_case(caller, case, http_mode=http_mode) for case in batch]
        )
        samples.extend(batch_samples)

    latencies = [sample.latency_ms for sample in samples]
    return {
        "samples": len(samples),
        "p50_ms": round(_percentile(latencies, 0.50), 2),
        "p80_ms": round(_percentile(latencies, 0.80), 2),
        "p95_ms": round(_percentile(latencies, 0.95), 2),
        "mean_ms": round(statistics.fmean(latencies), 2),
        "top_relation_hit_rate": round(
            sum(1 for sample in samples if sample.top_relation_hit) / max(len(samples), 1),
            4,
        ),
        "any_relation_hit_rate": round(
            sum(1 for sample in samples if sample.any_relation_hit) / max(len(samples), 1),
            4,
        ),
        "timing_breakdown_ms": _aggregate_timing_stats(samples),
        "counter_stats": _aggregate_counter_stats(samples),
        "flag_rates": _aggregate_flag_rates(samples),
        "cases": [asdict(sample) for sample in samples[: len(cases)]],
    }


async def _invoke_case(caller, case: BenchmarkCase, *, http_mode: bool) -> BenchmarkSample:
    started_at = perf_counter()
    response = await caller(case)
    latency_ms = (perf_counter() - started_at) * 1000.0

    if http_mode:
        response.raise_for_status()
        payload = response.json()
    else:
        payload = response.model_dump(mode="json")

    relations = payload.get("relations", [])
    diagnostics = payload.get("diagnostics") or {}
    top_relation_hit = bool(relations) and _relation_matches(relations[0], case)
    any_relation_hit = any(_relation_matches(relation, case) for relation in relations)
    return BenchmarkSample(
        case_name=case.name,
        latency_ms=latency_ms,
        top_relation_hit=top_relation_hit,
        any_relation_hit=any_relation_hit,
        result_count=len(payload.get("items", [])),
        relation_count=len(relations),
        timings_ms={key: float(value) for key, value in (diagnostics.get("timings_ms") or {}).items()},
        counters={key: int(value) for key, value in (diagnostics.get("counters") or {}).items()},
        flags={key: bool(value) for key, value in (diagnostics.get("flags") or {}).items()},
    )


def _relation_matches(relation: dict, case: BenchmarkCase) -> bool:
    if relation.get("relation_type") != case.expected_relation:
        return False
    fact = (relation.get("fact") or "").casefold()
    return all(term.casefold() in fact for term in case.expected_terms)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    index = max(min(int(round((len(sorted_values) - 1) * percentile)), len(sorted_values) - 1), 0)
    return sorted_values[index]


@contextmanager
def benchmark_workdir(data_dir: str | None):
    if data_dir:
        root = Path(data_dir)
        root.mkdir(parents=True, exist_ok=True)
        yield tempfile.mkdtemp(prefix="cosmic-memory-active-bench-", dir=str(root))
        return

    with tempfile.TemporaryDirectory(prefix="cosmic-memory-active-bench-") as tmp:
        yield tmp


def _aggregate_timing_stats(samples: list[BenchmarkSample]) -> dict[str, dict[str, float]]:
    keys = sorted({key for sample in samples for key in sample.timings_ms})
    stats: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [sample.timings_ms[key] for sample in samples if key in sample.timings_ms]
        stats[key] = {
            "mean": round(statistics.fmean(values), 3),
            "p50": round(_percentile(values, 0.50), 3),
            "p80": round(_percentile(values, 0.80), 3),
            "p95": round(_percentile(values, 0.95), 3),
        }
    return stats


def _aggregate_counter_stats(samples: list[BenchmarkSample]) -> dict[str, dict[str, float]]:
    keys = sorted({key for sample in samples for key in sample.counters})
    stats: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [float(sample.counters[key]) for sample in samples if key in sample.counters]
        stats[key] = {
            "mean": round(statistics.fmean(values), 3),
            "p50": round(_percentile(values, 0.50), 3),
            "p95": round(_percentile(values, 0.95), 3),
        }
    return stats


def _aggregate_flag_rates(samples: list[BenchmarkSample]) -> dict[str, float]:
    keys = sorted({key for sample in samples for key in sample.flags})
    rates: dict[str, float] = {}
    for key in keys:
        values = [sample.flags[key] for sample in samples if key in sample.flags]
        rates[key] = round(sum(1 for value in values if value) / max(len(values), 1), 4)
    return rates


async def async_main() -> None:
    args = parse_args()
    if args.mode == "http":
        report = await run_http_benchmark(args)
    else:
        report = await run_inprocess_benchmark(args)

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n", encoding="utf-8")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
