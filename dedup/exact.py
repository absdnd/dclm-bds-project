import hashlib
import time
import multiprocessing
from .base import Deduplicator
from .config import DedupConfig
import wandb
import pandas as pd
from collections import defaultdict
from .utils import log_duplicate_pair, save_duplicates  # Assuming you have these utils


class ExactHashDeduplicator(Deduplicator):
    def __init__(
        self,
        cfg: DedupConfig,
        debug_interval: int = 1000,  # how often to print progress
        log_duplicates: bool = True,
        top_n_duplicates: int = 10,
    ):
        self.cfg = cfg
        self.text_column = cfg.text_column
        self.debug_interval = debug_interval
        self.log_duplicates = log_duplicates
        self.top_n_duplicates = top_n_duplicates
        self.seen_hashes = set()
        self.hash_to_text = {}  # To store text for duplicate logging
        self.duplicate_groups = defaultdict(list)  # To track duplicate groups
        self.global_duplicates = 0
        self.num_process = cfg.num_process

    @staticmethod
    def _worker(args):
        """Worker only computes md5; does not touch shared state."""
        idx, text = args
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        return idx, h, text  # Now returning text as well for duplicate logging

    def run(self, examples: list[dict], step: int) -> list[dict]:
        unique_rows = []
        total = len(examples)
        start_time = time.time()
        duplicate_counts = [] if self.log_duplicates else None

        # Build the list of (index, text) payloads for workers
        tasks = [(i, ex[self.text_column]) for i, ex in enumerate(examples)]

        with multiprocessing.Pool(processes=self.num_process) as pool:
            for count, (idx, hash_val, text) in enumerate(
                pool.imap(self._worker, tasks), start=1
            ):
                if hash_val not in self.seen_hashes:
                    self.seen_hashes.add(hash_val)
                    self.hash_to_text[hash_val] = text
                    self.duplicate_groups[hash_val] = [(idx, text)]
                    unique_rows.append(examples[idx])
                    if self.log_duplicates:
                        duplicate_counts.append(0)
                else:
                    # Found duplicate
                    self.global_duplicates += 1
                    self.duplicate_groups[hash_val].append((idx, text))
                    if self.log_duplicates:
                        duplicate_counts.append(
                            len(self.duplicate_groups[hash_val]) - 1
                        )
                        # Log the duplicate pair
                        original_text = self.hash_to_text[hash_val]
                        log_duplicate_pair(
                            original_text=original_text,
                            duplicate_text=text,
                            threshold=1.0,  # Exact match has threshold of 1.0
                        )

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

        # Prepare metrics
        metrics = {
            "processed_total": total,
            "unique_docs": len(unique_rows),
            "duplicates": self.global_duplicates,
            "duplicate_ratio": self.global_duplicates / max(1, total),
            "top_duplicates": sorted(
                [
                    (len(items), items[0][1])
                    for items in self.duplicate_groups.values()
                    if len(items) > 1
                ],
                key=lambda x: x[0],
                reverse=True,
            )[: self.top_n_duplicates],
        }

        # Add duplicate counts to output documents if enabled
        if self.log_duplicates and duplicate_counts:
            for i, doc in enumerate(unique_rows):
                doc["duplicate_count"] = duplicate_counts[i]

        # Save duplicates to Excel
        if self.log_duplicates:
            save_duplicates(
                step=step
            )  # Assuming step is needed for your save_duplicates function
            # Optionally log to W&B
            # excel_path = save_duplicates()
            # wandb.log({"duplicates_file": wandb.Table(dataframe=pd.read_excel(excel_path))})

        return unique_rows, metrics
