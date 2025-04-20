import multiprocessing
import time
from datasketch import MinHash, MinHashLSH
from collections import defaultdict
from nltk.tokenize import word_tokenize

from .config import DedupConfig




from .base import Deduplicator


class MinHashDeduplicator(Deduplicator):
    
    def __init__(
        self,
        cfg: DedupConfig,
        debug_interval: int = 1000,  # how often to print debug
    ):
        """
        Initialize MinHash deduplicator with configuration.
        
        Args:
            cfg: DedupConfig object containing all configuration parameters
            debug_interval: How often to print debug information (default: 1000)
        """
        self.cfg = cfg
        self.text_column = cfg.text_column
        self.threshold = cfg.minhash_threshold
        self.num_hashes = cfg.minhash_num_hashes  
        self.debug_interval = debug_interval

        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_hashes)
        self.index = defaultdict(str)
        self.key_counter = 0


    @staticmethod
    def _worker(args):
        """
        Worker: build a MinHash from text tokens.
        args = (idx, text, num_hashes)
        """
        idx, text, num_hashes = args
        m = MinHash(num_perm=num_hashes)
        # use unique tokens for speed
        for token in set(word_tokenize(text)):
            m.update(token.encode("utf8"))
        return idx, m

    def run(self, examples: list[dict]) -> list[dict]:
        start_time = time.time()
        deduped = []
        total = len(examples)

        # prepare tasks for workers
        tasks = [
            (i, ex[self.text_column], self.num_hashes) for i, ex in enumerate(examples)
        ]

        with multiprocessing.Pool() as pool:
            for count, (idx, mh) in enumerate(pool.imap(self._worker, tasks), 1):
                ex = examples[idx]

                # sequential LSH query & insert
                if not self.lsh.query(mh):
                    key = f"doc_{self.key_counter}"
                    self.lsh.insert(key, mh)
                    self.index[key] = ex[self.text_column]
                    deduped.append(ex)
                    self.key_counter += 1

                # debug print every debug_interval (or at end)
                if count % self.debug_interval == 0 or count == total:
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else float("inf")
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"Processed {count}/{total} docs ― "
                        f"{rate:.1f} docs/sec, "
                        f"{len(deduped)} kept"
                    )

        return deduped
