
import multiprocessing
import time
import logging
from typing import Optional
from datasketch import MinHash, MinHashLSH
from collections import defaultdict
from nltk.tokenize import word_tokenize
import pandas as pd
import wandb

from .config import DedupConfig
from .base import Deduplicator




class MinHashDeduplicator(Deduplicator):
    
    def __init__(
        self,
        cfg: DedupConfig,
        debug_interval: int = 1000,  # how often to print debug
        log_duplicates: bool = True,
        top_n_duplicates: int = 10,
        logger: Optional[logging.Logger] = None
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
        self.log_duplicates = log_duplicates
        self.top_n_duplicates = getattr(cfg, 'top_n_duplicates', 10)
        self.use_wandb = getattr(cfg, 'use_wandb', True)

        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_hashes)
        self.index = defaultdict(list)  # Stores (chunk_id, doc_idx, text) tuples
        self.key_counter = 0
        self.processed_count = 0
        self.global_duplicates = 0

        


    @staticmethod
    def _worker(args):
        """
        Worker: build a MinHash from text tokens.
        args = (idx, text, num_hashes)
        """
        
        idx, text, num_hashes = args
        #print(text)
        m = MinHash(num_perm=num_hashes)
        # use unique tokens for speed
        for token in set(word_tokenize(text)):
            m.update(token.encode("utf8"))
        return idx, m

    def _log_metrics(self, chunk, deduped, chunk_duplicates):
        """Calculate and log metrics for the current chunk"""
        metrics = {
            'processed_total': self.processed_count,
            'chunk_size': len(chunk),
            'chunk_kept': len(deduped),
            'chunk_duplicates': chunk_duplicates,
            'global_unique': len(self.index),
            'global_duplicates': self.global_duplicates,
            'duplicate_ratio': self.global_duplicates / max(1, self.processed_count),
        }

        # Get top duplicates across all chunks
        top_duplicates = sorted(
            [(len(docs), docs[0][2]) for docs in self.index.values() if len(docs) > 1],
            key=lambda x: x[0],
            reverse=True
        )[:self.top_n_duplicates]

        if self.use_wandb:
            wandb.log(metrics)
            if top_duplicates:
                table = wandb.Table(columns=["Count", "Text Sample"], data=[
                    [count, text[:200] + "..." if len(text) > 200 else text]
                    for count, text in top_duplicates
                ])
                wandb.log({"top_duplicates": table})

        return metrics


    # def run(self, examples: list[dict]) -> list[dict]:
    #     start_time = time.time()
    #     deduped = []
    #     total = len(examples)

    #     # prepare tasks for workers
    #     tasks = [
    #         (i, ex[self.text_column], self.num_hashes) for i, ex in enumerate(examples)
    #     ]

    #     with multiprocessing.Pool() as pool:
    #         for count, (idx, mh) in enumerate(pool.imap(self._worker, tasks), 1):
    #             ex = examples[idx]

    #             # sequential LSH query & insert
    #             if not self.lsh.query(mh):
    #                 key = f"doc_{self.key_counter}"
    #                 self.lsh.insert(key, mh)
    #                 self.index[key] = ex[self.text_column]
    #                 deduped.append(ex)
    #                 self.key_counter += 1

    #             # debug print every debug_interval (or at end)
    #             if count % self.debug_interval == 0 or count == total:
    #                 elapsed = time.time() - start_time
    #                 rate = count / elapsed if elapsed > 0 else float("inf")
    #                 print(
    #                     f"[{time.strftime('%H:%M:%S')}] "
    #                     f"Processed {count}/{total} docs ― "
    #                     f"{rate:.1f} docs/sec, "
    #                     f"{len(deduped)} kept"
    #                 )

    #     return deduped

    # def run(self, examples: list[dict]) -> list[dict]:
    #     start_time = time.time()
    #     deduped = []
    #     duplicate_counts = [] if self.log_duplicates else None
    #     total = len(examples)
    #     self.processed_count += total

    #     # prepare tasks for workers
    #     tasks = [
    #         (i, ex[self.text_column], self.num_hashes) for i, ex in enumerate(examples)
    #     ]

    #     with multiprocessing.Pool() as pool:
    #         for count, (idx, mh) in enumerate(pool.imap(self._worker, tasks), 1):
    #             ex = examples[idx]
    #             text = ex[self.text_column]

    #             # Query for duplicates - This returns all LSH bucket keys that match the current document's MinHash (based on the Jaccard similarity threshold)
    #             matches = self.lsh.query(mh)
                
    #             #If no matches -> Creates new LSH bucket, starts new cluster with this document, sets duplicate count to 0

    #             if not matches:
    #                 # New unique document
    #                 key = f"doc_{self.key_counter}"
    #                 self.lsh.insert(key, mh)
    #                 self.index[key] = [idx]
    #                 deduped.append(ex)
    #                 if self.log_duplicates:
    #                     duplicate_counts.append(0)  # No duplicates
    #                 self.key_counter += 1
    #             else:

    #                 """
    #                 1. Adds document to existing cluster
    #                 2. Calculate duplicate count as cluster size -1
    #                 3. The document is not added to deduped list, only the first occurance will be there

    #                 """

    #                 # Found duplicates
    #                 matched_key = matches[0]  # Using first match
    #                 self.index[matched_key].append(idx)
    #                 if self.log_duplicates:
    #                     duplicate_count = len(self.index[matched_key]) - 1
    #                     duplicate_counts.append(duplicate_count)

    #             # debug print every debug_interval (or at end)
    #             if count % self.debug_interval == 0 or count == total:
    #                 elapsed = time.time() - start_time
    #                 rate = count / elapsed if elapsed > 0 else float("inf")
    #                 print(
    #                     f"[{time.strftime('%H:%M:%S')}] "
    #                     f"Processed {count}/{total} docs ― "
    #                     f"{rate:.1f} docs/sec, "
    #                     f"{len(deduped)} kept"
    #                 )

    #     # Calculate and log metrics
    #     metrics = self._calculate_metrics()
    #     print("\nDeduplication Metrics:")
    #     for k, v in metrics.items():
    #         print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    #     # Add duplicate counts if enabled
    #     if self.log_duplicates:
    #         for i, ex in enumerate(deduped):
    #             ex['duplicate_count'] = duplicate_counts[i]

    #     #self._upload_to_hf(deduped, duplicate_counts)

    #     return deduped

    def run(self, chunk: list[dict]) -> list[dict]:
        start_time = time.time()
        deduped = []
        chunk_duplicates = 0
        duplicate_counts = [] if self.log_duplicates else None
        chunk_id = id(chunk)  # Unique identifier for this chunk

        # Prepare parallel processing tasks
        tasks = [(i, doc[self.text_column], self.num_hashes) for i, doc in enumerate(chunk)]

        with multiprocessing.Pool() as pool:
            for count, (doc_idx, mh) in enumerate(pool.imap(self._worker, tasks), 1):
                doc = chunk[doc_idx]
                text = doc[self.text_column]

                # Query for duplicates
                matches = self.lsh.query(mh)
                
                if not matches:
                    # Unique document
                    key = f"doc_{self.key_counter}"
                    self.lsh.insert(key, mh)
                    self.index[key] = [(chunk_id, doc_idx, text)]
                    deduped.append(doc)
                    if self.log_duplicates:
                        duplicate_counts.append(0)
                    self.key_counter += 1
                else:
                    # Found duplicate
                    matched_key = matches[0]
                    self.index[matched_key].append((chunk_id, doc_idx, text))
                    chunk_duplicates += 1
                    self.global_duplicates += 1
                    if self.log_duplicates:
                        duplicate_counts.append(len(self.index[matched_key]) - 1)

                # Progress reporting
                if count % self.debug_interval == 0 or count == len(chunk):
                    elapsed = time.time() - start_time
                    rate = count / max(0.001, elapsed)
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Chunk {chunk_id} - "
                        f"Processed {count}/{len(chunk)} docs ({rate:.1f}/s) | "
                        f"Kept {len(deduped)} | Dupes {chunk_duplicates}"
                    )

        self.processed_count += len(chunk)
        metrics = self._log_metrics(chunk, deduped, chunk_duplicates)

        # Add duplicate counts if enabled
        if self.log_duplicates and duplicate_counts:
            for i, doc in enumerate(deduped):
                doc['duplicate_count'] = duplicate_counts[i]

        return deduped