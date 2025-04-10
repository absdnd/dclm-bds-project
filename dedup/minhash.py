from datasketch import MinHash, MinHashLSH
from nltk.tokenize import word_tokenize
from collections import defaultdict

from .base import Deduplicator


class MinHashDeduplicator(Deduplicator):
    def __init__(
        self,
        text_column: str = "text",
        threshold: float = 0.5,
        num_hashes: int = 120,
    ):
        """Initialize the MinHash deduplicator.

        Args:
            threshold (float): Jaccard similarity threshold for determining duplicates (0 to 1).
            num_hashes (int): Number of hash functions to use in MinHash.
        """
        self.text_column = text_column
        self.threshold = threshold
        self.num_hashes = num_hashes
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_hashes)
        self.index = defaultdict(list)
        self.key_counter = 0

    def _get_minhash(self, text: str) -> MinHash:
        """Convert text into a MinHash object."""
        minhash = MinHash(num_perm=self.num_hashes)
        tokens = word_tokenize(text)
        for t in tokens:
            minhash.update(t.encode("utf8"))  # Update MinHash with hashed tokens
        return minhash

    def run(self, examples: list[dict]) -> list[dict]:
        deduped = []
        for example in examples:
            text = example[self.text_column]
            minhash = self._get_minhash(text)
            matches = self.lsh.query(minhash)
            if not matches:
                key = f"doc_{self.key_counter}"
                self.lsh.insert(key, minhash)
                self.index[key] = text
                deduped.append(example)
                self.key_counter += 1
        return deduped
