from typing import Type
from .base import Deduplicator
from .exact import ExactHashDeduplicator
from .minhash import MinHashDeduplicator
from .bloomfilter import BloomFilterDeduplicator

DEDUPLICATOR_REGISTRY: dict[str, Type[Deduplicator]] = {
    "exact": ExactHashDeduplicator,
    "minhash": MinHashDeduplicator,
    "bloomfilter": BloomFilterDeduplicator,
}

