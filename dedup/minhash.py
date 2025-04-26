import nltk
nltk.download('punkt_tab')
import multiprocessing
from typing import Optional, Tuple, List, Dict
from datasketch import MinHash, MinHashLSH
from collections import defaultdict
from nltk.tokenize import word_tokenize

from .config import DedupConfig
from .base import Deduplicator

class MinHashDeduplicator(Deduplicator):
    
    def __init__(
        self,
        cfg: DedupConfig,
        debug_interval: int = 1000,
        log_duplicates: bool = True,
        top_n_duplicates: int = 10,

    ):
        self.cfg = cfg
        self.text_column = cfg.text_column
        self.threshold = cfg.minhash_threshold
        self.num_hashes = cfg.minhash_num_hashes  
        self.debug_interval = debug_interval
        self.log_duplicates = log_duplicates
        self.top_n_duplicates = top_n_duplicates

        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_hashes)
        self.index = defaultdict(list)
        self.key_counter = 0
        self.processed_count = 0
        self.global_duplicates = 0

    @staticmethod
    def _worker(args):
        """Worker: build a MinHash from text tokens."""
        idx, text, num_hashes = args
        m = MinHash(num_perm=num_hashes)
        for token in set(word_tokenize(text)):
            m.update(token.encode("utf8"))
        return idx, m

    def run(self, chunk: list[dict]) -> Tuple[List[dict], Dict]:
        """Run deduplication on a chunk of documents.
        
        Returns:
            - List of deduplicated documents
            - Dictionary of metrics for logging
        """
        deduped = []
        chunk_duplicates = 0
        duplicate_counts = [] if self.log_duplicates else None
        chunk_id = id(chunk)

        tasks = [(i, doc[self.text_column], self.num_hashes) for i, doc in enumerate(chunk)]

        with multiprocessing.Pool() as pool:
            for count, (doc_idx, mh) in enumerate(pool.imap(self._worker, tasks), 1):
                doc = chunk[doc_idx]
                text = doc[self.text_column]

                matches = self.lsh.query(mh)
                
                if not matches:
                    key = f"doc_{self.key_counter}"
                    self.lsh.insert(key, mh)
                    self.index[key] = [(chunk_id, doc_idx, text)]
                    deduped.append(doc)
                    if self.log_duplicates:
                        duplicate_counts.append(0)
                    self.key_counter += 1
                else:
                    matched_key = matches[0]
                    self.index[matched_key].append((chunk_id, doc_idx, text))
                    chunk_duplicates += 1
                    self.global_duplicates += 1
                    if self.log_duplicates:
                        duplicate_counts.append(len(self.index[matched_key]) - 1)

        self.processed_count += len(chunk)

        # Prepare metrics for logger
        metrics = {
            'processed_total': self.processed_count,
            'chunk_size': len(chunk),
            'chunk_kept': len(deduped),
            'chunk_duplicates': chunk_duplicates,
            'global_unique': len(self.index),
            'global_duplicates': self.global_duplicates,
            'duplicate_ratio': self.global_duplicates / max(1, self.processed_count),
            'top_duplicates': sorted(
                [(len(docs), docs[0][2]) for docs in self.index.values() if len(docs) > 1],
                key=lambda x: x[0],
                reverse=True
            )[:self.top_n_duplicates]
        }

        if self.log_duplicates and duplicate_counts:
            for i, doc in enumerate(deduped):
                doc['duplicate_count'] = duplicate_counts[i]

        return deduped, metrics