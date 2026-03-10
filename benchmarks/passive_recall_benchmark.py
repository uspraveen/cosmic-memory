"""Benchmark passive recall latency and result quality."""

from __future__ import annotations

import argparse
import asyncio
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
from cosmic_memory.embeddings.hash import HashEmbeddingService
from cosmic_memory.filesystem_service import FilesystemMemoryService
from cosmic_memory.graph import InMemoryGraphStore
from cosmic_memory.index.qdrant import (
    QdrantHybridMemoryIndex,
    _has_fastembed,
    _supports_qdrant_native_bm25,
)
from cosmic_memory.index.sparse import SimpleSparseEncoder


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["inprocess", "http"], default="inprocess")
    parser.add_argument("--base-url", default=None, help="Required for HTTP mode.")
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
    graph_store = InMemoryGraphStore() if args.graph_backend == "memory" else None
    with tempfile.TemporaryDirectory(prefix="cosmic-memory-bench-", dir=args.data_dir) as tmp:
        if _supports_qdrant_native_bm25() and _has_fastembed():
            index = QdrantHybridMemoryIndex(
                embedding_service=HashEmbeddingService(dimensions=256),
                path=str(Path(tmp) / "qdrant"),
                vector_size=256,
            )
        else:
            index = QdrantHybridMemoryIndex(
                embedding_service=HashEmbeddingService(dimensions=256),
                sparse_encoder=SimpleSparseEncoder(),
                path=str(Path(tmp) / "qdrant"),
                vector_size=256,
            )
        service = FilesystemMemoryService(
            tmp,
            passive_index=index,
            graph_store=graph_store,
            passive_graph_timeout_seconds=args.graph_timeout_ms / 1000.0,
        )
        await seed_inprocess_service(service, args.records)
        report = await _exercise_passive_recall(
            lambda case: service.passive_recall(
                PassiveRecallRequest(
                    query=case.query,
                    max_results=case.max_results,
                    token_budget=case.token_budget,
                )
            ),
            warmup=args.warmup,
            iterations=args.iterations,
            concurrency=args.concurrency,
        )
        await index.close()
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
