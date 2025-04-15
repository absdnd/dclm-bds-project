from dedup.base import Deduplicator
import hashlib

'''
A Bloom filter is a space-efficient probabilistic data structure that is used to test whether an item is a member of a set. The bloom filter will always say yes if an item is a set member. However, the bloom filter might still say yes although an item is not a member of the set (false positive). The items can be added to the bloom filter but the items cannot be removed. The bloom filter supports the following operations:

    adding an item to the set
    test the membership of an item in the set
Source: https://systemdesign.one/bloom-filters-explained/
'''

class BloomFilterDeduplicator(Deduplicator):
    def __init__(self, capacity: int = 100000, error_rate: float = 0.001, key: str = None):
        """
        :param capacity: Expected number of unique items.
        :param error_rate: Acceptable false positive rate.
        :param key: Optional key in dict to hash (if None, use entire dict).
        """
        self.bloom = BloomFilterDeduplicator(capacity=capacity, error_rate=error_rate)
        self.key = key

    def _hash(self, example: dict) -> str:
        if self.key is not None and self.key in example:
            content = str(example[self.key])
        else:
            content = str(sorted(example.items()))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


    def run(self, examples: list[dict]) -> list[dict]:
        deduped_examples = []
        for example in examples:
            fingerprint = self._hash(example)
            # checking membership
            if fingerprint not in self.bloom:
                # adding member to set
                self.bloom.add(fingerprint)
                deduped_examples.append(example)
        return deduped_examples