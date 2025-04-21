
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
        self.top_n_duplicates = top_n_duplicates if hasattr(cfg, 'top_n_duplicates') else 10
        self.use_wandb = True
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_hashes)
        # stores list of doc indices for each key - maps LSH keys to list of document indices
        #Each entry in index represents a cluster of similar documents
        self.index = defaultdict(list)
    
        self.key_counter = 0
        self.duplicate_counts = defaultdict(int)
        self.processed_count = 0
        self.duplicate_stats = {
            'total_duplicates': 0,
            'unique_docs': 0,
            'duplicate_clusters': 0
        }

        


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

    def _calculate_metrics(self):
        """Calculate and log deduplication metrics"""
        # Calculate various metrics
        duplicate_counts = [len(docs) for docs in self.index.values() if len(docs) > 1]
        
        metrics = {
            'total_docs_processed': self.processed_count,
            'unique_docs': len(self.index),
            'total_duplicates_found': self.processed_count - len(self.index),
            'duplicate_ratio': (self.processed_count - len(self.index)) / max(1, self.processed_count),
            'avg_duplicates_per_cluster': sum(duplicate_counts) / max(1, len(duplicate_counts)),
            'max_duplicates_in_cluster': max(duplicate_counts) if duplicate_counts else 0,
        }

        # Get top N duplicates
        top_duplicates = sorted(
            [(len(docs), docs[0]) for docs in self.index.values() if len(docs) > 1],
            key=lambda x: x[0],
            reverse=True
        )[:self.top_n_duplicates]

        # Log to WandB
        if self.use_wandb:
            wandb.log(metrics)
            
            # Create a table of top duplicates for WandB
            if top_duplicates:
                top_dup_table = wandb.Table(columns=["Count", "Example Text"], data=[
                    [count, text[:200] + "..." if len(text) > 200 else text] 
                    for count, text in top_duplicates
                ])
                wandb.log({"Top Duplicates": top_dup_table})

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

    def run(self, examples: list[dict]) -> list[dict]:
        start_time = time.time()
        deduped = []
        duplicate_counts = [] if self.log_duplicates else None
        total = len(examples)
        self.processed_count += total

        # prepare tasks for workers
        tasks = [
            (i, ex[self.text_column], self.num_hashes) for i, ex in enumerate(examples)
        ]

        with multiprocessing.Pool() as pool:
            for count, (idx, mh) in enumerate(pool.imap(self._worker, tasks), 1):
                ex = examples[idx]
                text = ex[self.text_column]

                # Query for duplicates - This returns all LSH bucket keys that match the current document's MinHash (based on the Jaccard similarity threshold)
                matches = self.lsh.query(mh)
                
                #If no matches -> Creates new LSH bucket, starts new cluster with this document, sets duplicate count to 0

                if not matches:
                    # New unique document
                    key = f"doc_{self.key_counter}"
                    self.lsh.insert(key, mh)
                    self.index[key] = [idx]
                    deduped.append(ex)
                    if self.log_duplicates:
                        duplicate_counts.append(0)  # No duplicates
                    self.key_counter += 1
                else:

                    """
                    1. Adds document to existing cluster
                    2. Calculate duplicate count as cluster size -1
                    3. The document is not added to deduped list, only the first occurance will be there

                    """

                    # Found duplicates
                    matched_key = matches[0]  # Using first match
                    self.index[matched_key].append(idx)
                    if self.log_duplicates:
                        duplicate_count = len(self.index[matched_key]) - 1
                        duplicate_counts.append(duplicate_count)

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

        # Calculate and log metrics
        metrics = self._calculate_metrics()
        print("\nDeduplication Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        # Add duplicate counts if enabled
        if self.log_duplicates:
            for i, ex in enumerate(deduped):
                ex['duplicate_count'] = duplicate_counts[i]

        #self._upload_to_hf(deduped, duplicate_counts)

        return deduped