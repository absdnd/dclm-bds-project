'''
A Bloom filter is a space-efficient probabilistic data structure that is used to test whether an item is a member of a set. The bloom filter will always say yes if an item is a set member. However, the bloom filter might still say yes although an item is not a member of the set (false positive). The items can be added to the bloom filter but the items cannot be removed. The bloom filter supports the following operations:

    adding an item to the set
    test the membership of an item in the set
Source: https://systemdesign.one/bloom-filters-explained/
'''

from dedup.base import Deduplicator
import hashlib
import math
import bitarray

'''
More aggressive dedupe:
Lower error_rate → More accurate, fewer false positives → more correct duplicates removed.
Increase capacity → Reduces collision, better performance.

Less aggressive dedupe:
Increase error rate or reduce capacity
'''

class SimpleBloomFilter:
    def __init__(self, capacity: int, error_rate: float):
        self.capacity = capacity
        self.error_rate = error_rate
        self.size = self._get_size(capacity, error_rate)
        self.hash_count = self._get_hash_count(self.size, capacity)
        self.bit_array = bitarray.bitarray(self.size)
        self.bit_array.setall(False)

    def _get_size(self, n: int, p: float) -> int:
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    def _get_hash_count(self, m: int, n: int) -> int:
        k = (m / n) * math.log(2)
        return int(k)

    def _hashes(self, item: str):
        for i in range(self.hash_count):
            yield int(hashlib.sha256((item + str(i)).encode("utf-8")).hexdigest(), 16) % self.size

    def add(self, item: str):
        for i in self._hashes(item):
            self.bit_array[i] = True

    def __contains__(self, item: str):
        return all(self.bit_array[i] for i in self._hashes(item))


class BloomFilterDeduplicator(Deduplicator):
    def __init__(self, 
                 text_column: str = "text",
                 capacity: int = 100000, 
                 error_rate: float = 0.001, 
                 key: str = None):
        self.text_column = text_column
        self.bloom = SimpleBloomFilter(capacity=capacity, error_rate=error_rate)
        self.key = key

    def _hash(self, example: dict) -> str:
        if self.key is not None and self.key in example:
            content = str(example[self.key])
        else:
            content = str(example.get(self.text_column, ""))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def run(self, examples: list[dict]) -> list[dict]:
        deduped_examples = []
        for example in examples:
            fingerprint = self._hash(example)
            if fingerprint not in self.bloom:
                self.bloom.add(fingerprint)
                deduped_examples.append(example)
        return deduped_examples

