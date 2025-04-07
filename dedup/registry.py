from typing import Type
from .base import Deduplicator
from .exact import ExactHashDeduplicator
from .minhash import MinHashDeduplicator

DEDUPLICATOR_REGISTRY: dict[str, Type[Deduplicator]] = {
    "exact": ExactHashDeduplicator,
    "minhash": MinHashDeduplicator,
}
