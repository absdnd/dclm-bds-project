import hashlib
import time
import multiprocessing
from .base import Deduplicator


class ExactHashDeduplicator(Deduplicator):
    def __init__(
        self,
        text_column: str = "text",
        debug_interval: int = 1000,  # how often to print progress
    ):
        self.text_column = text_column
        self.debug_interval = debug_interval
        # This is the single, shared set—only the main process will update it
        self.seen_hashes = set()

    @staticmethod
    def _worker(args):
        """Worker only computes md5; does not touch shared state."""
        idx, text = args
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        return idx, h

    def run(self, examples: list[dict]) -> list[dict]:
        unique_rows = []
        total = len(examples)
        start_time = time.time()

        # Build the list of (index, text) payloads for workers
        tasks = [(i, ex[self.text_column]) for i, ex in enumerate(examples)]

        with multiprocessing.Pool() as pool:
            # imap preserves order; you could use imap_unordered if order doesn't matter
            for count, (idx, hash_val) in enumerate(
                pool.imap(self._worker, tasks), start=1
            ):
                # MAIN PROCESS: do the dedup logic here
                if hash_val not in self.seen_hashes:
                    self.seen_hashes.add(hash_val)
                    unique_rows.append(examples[idx])

                # debug print every debug_interval or at the end
                if count % self.debug_interval == 0 or count == total:
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else float("inf")
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"Processed {count}/{total} docs ― "
                        f"{rate:.1f} docs/sec, "
                        f"{len(unique_rows)} unique"
                    )

        return unique_rows
