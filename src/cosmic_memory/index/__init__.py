"""Passive recall index adapters."""

from cosmic_memory.index.base import PassiveMemoryIndex
from cosmic_memory.index.null import NullPassiveMemoryIndex
from cosmic_memory.index.qdrant import QdrantHybridMemoryIndex
from cosmic_memory.index.sparse import FastEmbedSparseEncoder, SimpleSparseEncoder, SparseEncoder

__all__ = [
    "FastEmbedSparseEncoder",
    "NullPassiveMemoryIndex",
    "PassiveMemoryIndex",
    "QdrantHybridMemoryIndex",
    "SimpleSparseEncoder",
    "SparseEncoder",
]
