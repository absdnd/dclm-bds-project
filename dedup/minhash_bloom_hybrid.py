from .base import Deduplicator
import hashlib
import time
import multiprocessing
from .config import DedupConfig
import wandb
from datasketch import MinHash
from .bloomfilter import SimpleBloomFilter


def get_shingles(text: str, n: int = 5) -> set:
    tokens = text.split()
    return set([" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])


def compute_minhash_signature(shingles: set, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in shingles:
        m.update(shingle.encode("utf-8"))
    return m


def hash_signature(mh: MinHash) -> str:
    return hashlib.sha256(mh.digest()).hexdigest()


class MinHashBloomDeduplicator(Deduplicator):
    def __init__(self, cfg: DedupConfig, key: str | None = None):
        self.cfg = cfg
        self.text_column = cfg.text_column
        self.key = key
        self.debug_interval = cfg.bloom_debug_interval
        self.num_process = cfg.num_process

        self.bloom = SimpleBloomFilter(
            capacity=cfg.bloom_capacity,
            error_rate=cfg.bloom_error_rate
        )

        self.ngram_size = cfg.minhash_ngram_size
        self.num_perm = cfg.minhash_num_perm

    @staticmethod
    def _worker(args):
        idx, example, text_column, key, ngram_size, num_perm = args
        content = str(example.get(key if key else text_column, ""))
        shingles = get_shingles(content, n=ngram_size)
        signature = compute_minhash_signature(shingles, num_perm=num_perm)
        fingerprint = hash_signature(signature)
        return idx, fingerprint

    def run(self, examples: list[dict], steps: int) -> list[dict]:
        total = len(examples)
        start = time.time()
        deduped = []
        saturation_check_interval = 10000

        tasks = [
            (i, ex, self.text_column, self.key, self.ngram_size, self.num_perm)
            for i, ex in enumerate(examples)
        ]

        with multiprocessing.Pool(processes=self.num_process) as pool:
            for count, (idx, fingerprint) in enumerate(pool.imap(self._worker, tasks), start=1):
                if fingerprint not in self.bloom:
                    self.bloom.add(fingerprint)
                    deduped.append(examples[idx])

                if count % self.debug_interval == 0 or count == total:
                    elapsed = time.time() - start
                    rate = count / elapsed if elapsed > 0 else float("inf")
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"Processed {count}/{total} docs ― "
                        f"{rate:.1f} docs/sec, "
                        f"{len(deduped)} unique"
                    )

                if count % saturation_check_interval == 0:
                    saturation = self.bloom.bit_array.count(True) / self.bloom.size
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"Bloom filter saturation: {saturation:.4f} ({saturation * 100:.2f}%)"
                    )
                    wandb.log({
                        "bloom_saturation": saturation,
                        "bloom_saturation_percent": saturation * 100,
                        "processed_docs": count,
                    }, step=steps)

        metrics = {}
        return deduped, metrics