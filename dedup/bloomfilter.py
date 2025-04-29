from .base import Deduplicator
import hashlib
import math
import bitarray
import time
import multiprocessing
from .config import DedupConfig


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
            yield int(
                hashlib.sha256((item + str(i)).encode("utf-8")).hexdigest(),
                16,
            ) % self.size

    def add(self, item: str):
        for i in self._hashes(item):
            self.bit_array[i] = True

    def __contains__(self, item: str):
        return all(self.bit_array[i] for i in self._hashes(item))


class BloomFilterDeduplicator(Deduplicator):
    def __init__(
        self,
        cfg: DedupConfig,
        key: str | None = None,
    ):
        self.cfg = cfg
        self.text_column = cfg.text_column
        self.bloom = SimpleBloomFilter(
            capacity=cfg.bloom_capacity, error_rate=cfg.bloom_error_rate
        )
        self.key = key
        self.debug_interval = cfg.bloom_debug_interval
        self.num_process = cfg.num_process

    @staticmethod
    def _worker(args):
        """
        Worker only computes the fingerprint; does NOT touch bloom.
        Returns (idx, fingerprint).
        """
        idx, example, text_column, key = args
        if key is not None and key in example:
            content = str(example[key])
        else:
            content = str(example.get(text_column, ""))
        fp = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return idx, fp

    def run(self, examples: list[dict], steps: int) -> list[dict]:
        total = len(examples)
        start = time.time()
        deduped = []

        # prepare arguments for each worker
        tasks = [(i, ex, self.text_column, self.key) for i, ex in enumerate(examples)]

        with multiprocessing.Pool(processes=self.num_process) as pool:
            for count, (idx, fingerprint) in enumerate(
                pool.imap(self._worker, tasks), start=1
            ):
                # MAIN PROCESS: consistent bloom membership + insert
                if fingerprint not in self.bloom:
                    self.bloom.add(fingerprint)
                    deduped.append(examples[idx])

                # debug print
                if count % self.debug_interval == 0 or count == total:
                    elapsed = time.time() - start
                    rate = count / elapsed if elapsed > 0 else float("inf")
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"Processed {count}/{total} docs ― "
                        f"{rate:.1f} docs/sec, "
                        f"{len(deduped)} unique"
                    )

        metrics = {}
        return deduped, metrics
