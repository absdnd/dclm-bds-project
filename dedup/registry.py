from typing import Type
from .base import Deduplicator
from .exact import ExactHashDeduplicator
from .minhash import MinHashDeduplicator
from .bloomfilter import BloomFilterDeduplicator
from .minhash_bloom_hybrid import MinHashBloomDeduplicator

DEDUPLICATOR_REGISTRY: dict[str, Type[Deduplicator]] = {
    "exact": ExactHashDeduplicator,
    "minhash": MinHashDeduplicator,
    "bloomfilter": BloomFilterDeduplicator,
    "minhash_bloom_hybrid": MinHashBloomDeduplicator,
}

