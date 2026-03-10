"""Benchmark passive recall latency and result quality."""

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
from cosmic_memory.domain.models import MemoryProvenance, PassiveRecallRequest, WriteMemoryRequest
from cosmic_memory.embeddings import PerplexityStandardEmbeddingService
from cosmic_memory.embeddings.base import EmbeddingService
from cosmic_memory.embeddings.hash import HashEmbeddingService
from cosmic_memory.env import load_env_file
from cosmic_memory.filesystem_service import FilesystemMemoryService
from cosmic_memory.graph import InMemoryGraphStore
from cosmic_memory.index.qdrant import (
    QdrantHybridMemoryIndex,
    _has_fastembed,
    _supports_qdrant_native_bm25,
)
from cosmic_memory.index.sparse import FastEmbedSparseEncoder, SimpleSparseEncoder


@dataclass(slots=True)
class BenchmarkCase:
    name: str
    query: str
    expected_kind: MemoryKind
    expected_terms: tuple[str, ...]
    token_budget: int = 800
    max_results: int = 6


@dataclass(slots=True)
class BenchmarkSample:
    case_name: str
    latency_ms: float
    top1_hit: bool
    any_hit: bool
    total_token_count: int
    result_count: int
    within_budget: bool
    timings_ms: dict[str, float]
    counters: dict[str, int]
    flags: dict[str, bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["inprocess", "http"], default="inprocess")
    parser.add_argument("--base-url", default=None, help="Required for HTTP mode.")
    parser.add_argument(
        "--embedding-backend",
        choices=["perplexity", "hash"],
        default="perplexity",
    )
    parser.add_argument("--embedding-model", default="pplx-embed-v1-4b")
    parser.add_argument("--embedding-dimensions", type=int, default=1024)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--embed-max-parallel", type=int, default=4)
    parser.add_argument("--records", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--graph-backend", choices=["none", "memory"], default="memory")
    parser.add_argument("--graph-timeout-ms", type=int, default=120)
    parser.add_argument("--data-dir", default=None, help="Optional benchmark data directory.")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def provenance() -> MemoryProvenance:
    return MemoryProvenance(source_kind="benchmark", created_by="passive_recall_benchmark")


def benchmark_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            name="core_fact_preference",
            query="What are the user's concise answer preferences?",
            expected_kind=MemoryKind.CORE_FACT,
            expected_terms=("concise", "answers"),
        ),
        BenchmarkCase(
            name="identity_relationship",
            query="What project does nxagarwal@ualr.edu work on?",
            expected_kind=MemoryKind.AGENT_NOTE,
            expected_terms=("cosmic", "memory"),
        ),
        BenchmarkCase(
            name="current_state",
            query="What is the current active project?",
            expected_kind=MemoryKind.AGENT_NOTE,
            expected_terms=("current", "active", "cosmic", "memory"),
        ),
        BenchmarkCase(
            name="decision_memory",
            query="What did we decide about Qdrant BM25 retrieval?",
            expected_kind=MemoryKind.SESSION_SUMMARY,
            expected_terms=("qdrant", "bm25"),
        ),
        BenchmarkCase(
            name="latency_budget",
            query="What latency target are we aiming for in passive recall?",
            expected_kind=MemoryKind.TASK_SUMMARY,
            expected_terms=("300", "400", "ms"),
        ),
    ]


def synthetic_requests(total_records: int) -> list[WriteMemoryRequest]:
    fixed = [
        WriteMemoryRequest(
            kind=MemoryKind.CORE_FACT,
            title="Response preferences",
            content="User prefers concise answers, direct explanations, and minimal filler.",
            tags=["preference", "current", "active"],
            provenance=provenance(),
        ),
        WriteMemoryRequest(
            kind=MemoryKind.AGENT_NOTE,
            title="Research profile",
            content="Nitin Agarwal works on Cosmic Memory and graph retrieval.",
            tags=["profile", "relationship"],
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
            kind=MemoryKind.AGENT_NOTE,
            title="Current focus",
            content="The current active project is Cosmic Memory and it remains the main focus.",
            tags=["current", "active", "focus"],
            provenance=provenance(),
            metadata={
                "graph_document": {
                    "memory_id": "ignored",
                    "entities": [
                        {
                            "local_ref": "project",
                            "entity_type": "project",
                            "canonical_name": "Cosmic Memory",
                        }
                    ],
                    "relations": [],
                }
            },
        ),
        WriteMemoryRequest(
            kind=MemoryKind.SESSION_SUMMARY,
            title="Hybrid retrieval decision",
            content="We decided to keep Qdrant native BM25 plus dense retrieval for passive recall.",
            tags=["architecture", "qdrant", "bm25"],
            provenance=provenance(),
        ),
        WriteMemoryRequest(
            kind=MemoryKind.TASK_SUMMARY,
            title="Passive recall targets",
            content="Passive recall should stay around 300 to 400 ms in at least 80 percent of requests.",
            tags=["latency", "budget"],
            provenance=provenance(),
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
    kinds = [
        MemoryKind.USER_DATA,
        MemoryKind.SESSION_SUMMARY,
        MemoryKind.TASK_SUMMARY,
        MemoryKind.AGENT_NOTE,
    ]
    filler_count = max(total_records - len(fixed), 0)
    for index in range(filler_count):
        topic = filler_topics[index % len(filler_topics)]
        noise = " ".join(randomizer.sample(filler_topics, k=min(4, len(filler_topics))))
        fixed.append(
            WriteMemoryRequest(
                kind=kinds[index % len(kinds)],
                title=f"Synthetic record {index + 1}",
                content=f"Synthetic memory about {topic}. Related notes: {noise}.",
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
    graph_store = InMemoryGraphStore() if args.graph_backend == "memory" else None
    embedding_service = build_embedding_service(args)
    with benchmark_workdir(args.data_dir) as tmp:
        if _has_fastembed():
            index = QdrantHybridMemoryIndex(
                embedding_service=embedding_service,
                sparse_encoder=FastEmbedSparseEncoder(),
                path=str(Path(tmp) / "qdrant"),
                vector_size=embedding_service.dimensions,
            )
        elif _supports_qdrant_native_bm25():
            index = QdrantHybridMemoryIndex(
                embedding_service=embedding_service,
                path=str(Path(tmp) / "qdrant"),
                vector_size=embedding_service.dimensions,
            )
        else:
            index = QdrantHybridMemoryIndex(
                embedding_service=embedding_service,
                sparse_encoder=SimpleSparseEncoder(),
                path=str(Path(tmp) / "qdrant"),
                vector_size=embedding_service.dimensions,
            )
        service = FilesystemMemoryService(
            tmp,
            passive_index=None,
            graph_store=graph_store,
            passive_graph_timeout_seconds=args.graph_timeout_ms / 1000.0,
        )
        await seed_inprocess_service(service, args.records)
        service.passive_index = index
        await service.sync_index()
        report = await _exercise_passive_recall(
            lambda case: service.passive_recall(
                PassiveRecallRequest(
                    query=case.query,
                    max_results=case.max_results,
                    token_budget=case.token_budget,
                    include_diagnostics=True,
                )
            ),
            warmup=args.warmup,
            iterations=args.iterations,
            concurrency=args.concurrency,
        )
        await index.close()
        await embedding_service.close()
        return report


async def run_http_benchmark(args: argparse.Namespace) -> dict:
    if not args.base_url:
        raise SystemExit("--base-url is required when --mode=http")
    async with httpx.AsyncClient(base_url=args.base_url, timeout=30.0) as client:
        await seed_http_service(client, args.records)
        return await _exercise_passive_recall(
            lambda case: client.post(
                "/v1/query/passive",
                json=PassiveRecallRequest(
                    query=case.query,
                    max_results=case.max_results,
                    token_budget=case.token_budget,
                    include_diagnostics=True,
                ).model_dump(mode="json"),
            ),
            warmup=args.warmup,
            iterations=args.iterations,
            concurrency=args.concurrency,
            http_mode=True,
        )


async def _exercise_passive_recall(
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
    report = {
        "samples": len(samples),
        "p50_ms": round(_percentile(latencies, 0.50), 2),
        "p80_ms": round(_percentile(latencies, 0.80), 2),
        "p95_ms": round(_percentile(latencies, 0.95), 2),
        "mean_ms": round(statistics.fmean(latencies), 2),
        "top1_hit_rate": round(
            sum(1 for sample in samples if sample.top1_hit) / max(len(samples), 1), 4
        ),
        "any_hit_rate": round(
            sum(1 for sample in samples if sample.any_hit) / max(len(samples), 1), 4
        ),
        "budget_compliance_rate": round(
            sum(1 for sample in samples if sample.within_budget) / max(len(samples), 1),
            4,
        ),
        "timing_breakdown_ms": _aggregate_timing_stats(samples),
        "counter_stats": _aggregate_counter_stats(samples),
        "flag_rates": _aggregate_flag_rates(samples),
        "cases": [asdict(sample) for sample in samples[: len(cases)]],
        "budget_ok_samples": sum(1 for sample in samples if sample.within_budget),
    }
    return report


async def _invoke_case(caller, case: BenchmarkCase, *, http_mode: bool) -> BenchmarkSample:
    started_at = perf_counter()
    response = await caller(case)
    latency_ms = (perf_counter() - started_at) * 1000.0

    if http_mode:
        response.raise_for_status()
        payload = response.json()
    else:
        payload = response.model_dump(mode="json")

    items = payload.get("items", [])
    diagnostics = payload.get("diagnostics") or {}
    top1_hit = bool(items) and _item_matches(items[0], case)
    any_hit = any(_item_matches(item, case) for item in items)
    return BenchmarkSample(
        case_name=case.name,
        latency_ms=latency_ms,
        top1_hit=top1_hit,
        any_hit=any_hit,
        total_token_count=int(payload.get("total_token_count", 0)),
        result_count=len(items),
        within_budget=int(payload.get("total_token_count", 0)) <= case.token_budget,
        timings_ms={key: float(value) for key, value in (diagnostics.get("timings_ms") or {}).items()},
        counters={key: int(value) for key, value in (diagnostics.get("counters") or {}).items()},
        flags={key: bool(value) for key, value in (diagnostics.get("flags") or {}).items()},
    )


def _item_matches(item: dict, case: BenchmarkCase) -> bool:
    if item.get("kind") != case.expected_kind.value:
        return False
    text = " ".join([item.get("title") or "", item.get("content") or ""]).casefold()
    return all(term.casefold() in text for term in case.expected_terms)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    index = max(min(int(round((len(sorted_values) - 1) * percentile)), len(sorted_values) - 1), 0)
    return sorted_values[index]


def build_embedding_service(args: argparse.Namespace) -> EmbeddingService:
    if args.embedding_backend == "hash":
        return HashEmbeddingService(dimensions=args.embedding_dimensions)

    return PerplexityStandardEmbeddingService(
        model_name=args.embedding_model,
        dimensions=args.embedding_dimensions,
        batch_size=args.embed_batch_size,
        max_parallel_requests=args.embed_max_parallel,
    )


@contextmanager
def benchmark_workdir(data_dir: str | None):
    if data_dir:
        root = Path(data_dir)
        root.mkdir(parents=True, exist_ok=True)
        yield tempfile.mkdtemp(prefix="cosmic-memory-bench-", dir=str(root))
        return

    with tempfile.TemporaryDirectory(prefix="cosmic-memory-bench-") as tmp:
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
