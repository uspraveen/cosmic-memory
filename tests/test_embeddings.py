import asyncio
import base64
import math
from array import array
from types import SimpleNamespace

from cosmic_memory.domain.models import GenerateEmbeddingsRequest
from cosmic_memory.embeddings.hash import HashEmbeddingService
from cosmic_memory.embeddings.perplexity import PerplexityStandardEmbeddingService


def test_hash_embedding_service_generates_requested_dimensions():
    async def run():
        service = HashEmbeddingService(dimensions=64)
        response = await service.generate(
            GenerateEmbeddingsRequest(texts=["Cosmic memory"], dimensions=128)
        )

        assert response.model == "hash-embedding-dev"
        assert response.dimensions == 128
        assert len(response.items) == 1
        assert response.items[0].dimensions == 128
        assert len(response.items[0].vector) == 128

    asyncio.run(run())


def test_perplexity_embedding_service_batches_and_aggregates_usage():
    class FakeEmbeddingsResource:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            dimensions = kwargs["dimensions"]
            data = [
                SimpleNamespace(
                    index=index,
                    embedding=_encoded_int8_vector(seed=index + len(self.calls), dimensions=dimensions),
                )
                for index, _text in enumerate(kwargs["input"])
            ]
            return SimpleNamespace(
                data=data,
                usage=SimpleNamespace(
                    prompt_tokens=len(kwargs["input"]),
                    total_tokens=len(kwargs["input"]),
                    cost=SimpleNamespace(
                        currency="USD",
                        input_cost=0.1 * len(kwargs["input"]),
                        total_cost=0.2 * len(kwargs["input"]),
                    ),
                ),
            )

    class FakeClient:
        def __init__(self) -> None:
            self.embeddings = FakeEmbeddingsResource()

        async def close(self) -> None:
            return None

    async def run():
        client = FakeClient()
        service = PerplexityStandardEmbeddingService(
            client=client,
            dimensions=8,
            batch_size=2,
            max_parallel_requests=2,
        )

        response = await service.generate(
            GenerateEmbeddingsRequest(
                texts=["alpha", "beta", "gamma"],
                dimensions=128,
                batch_size=2,
                max_parallel_requests=2,
            )
        )

        assert response.model == "pplx-embed-v1-4b"
        assert response.dimensions == 128
        assert [item.index for item in response.items] == [0, 1, 2]
        assert len(client.embeddings.calls) == 2
        assert client.embeddings.calls[0]["dimensions"] == 128
        assert response.usage is not None
        assert response.usage.prompt_tokens == 3
        assert response.usage.total_tokens == 3
        assert response.usage.cost is not None
        assert math.isclose(response.usage.cost.input_cost or 0.0, 0.3)
        assert math.isclose(response.usage.cost.total_cost or 0.0, 0.6)
        for item in response.items:
            assert len(item.vector) == 128
            norm = math.sqrt(sum(value * value for value in item.vector))
            assert math.isclose(norm, 1.0, rel_tol=1e-6)

    asyncio.run(run())


def _encoded_int8_vector(*, seed: int, dimensions: int) -> str:
    values = array(
        "b",
        [((seed + index) % 11) - 5 for index in range(dimensions)],
    )
    return base64.b64encode(values.tobytes()).decode("ascii")
