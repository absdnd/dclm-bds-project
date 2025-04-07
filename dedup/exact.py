import hashlib

from .base import Deduplicator


class ExactHashDeduplicator(Deduplicator):
    def __init__(self, seen_hashes: set[str], text_column: str = "text"):
        self.text_column = text_column
        self.seen_hashes = seen_hashes if seen_hashes is not None else set()

    def run(self, examples: list[dict]) -> list[dict]:
        unique_rows = []
        for row in examples:
            text = row[self.text_column]
            hash_val = hashlib.md5(text.encode("utf-8")).hexdigest()
            if hash_val not in self.seen_hashes:
                self.seen_hashes.add(hash_val)
                unique_rows.append(row)
        return unique_rows
