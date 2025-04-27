import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import gc
from typing import Dict, Any, List, Optional

class MetricsLogger:
    def __init__(self):
        # Core metrics tracking
        self.chunk_indices = []
        self.original_counts = []
        self.deduped_counts = []
        
        # Performance metrics
        self.total_runtime = 0.0
        self.total_memory_mb = 0.0

        self.similarity_scores_before = []
        self.similarity_scores_after = []
        
        # Deduplication metrics storage
        self.dedup_metrics = {
            'processed_total': [],
            'duplicate_ratio': [],
            'global_unique': [],
            'global_duplicates': []
        }

    def run_with_metrics(self, fn, *args, **kwargs) -> tuple[Any, Dict[str, Any]]:
        """Measure runtime and memory usage of a function."""
        gc.collect()
        start_time = time.time()
        process = psutil.Process()
        start_mem = process.memory_info().rss

        result = fn(*args, **kwargs)

        end_time = time.time()
        end_mem = process.memory_info().rss

        runtime = end_time - start_time
        mem_used = (end_mem - start_mem) / 1024 / 1024  # in MB
        
        self.total_memory_mb += mem_used
        self.total_runtime += runtime
        
        metrics = {
            'runtime_sec': runtime,
            'memory_usage_mb': mem_used
        }
        return result, metrics

    def log_chunk_metrics(self, 
                         chunk_index: int, 
                         original_len: int, 
                         deduped_len: int,
                         runtime_mem_metrics: Dict[str, float]):
        """Log metrics for a processed chunk."""
        self.chunk_indices.append(chunk_index)
        self.original_counts.append(original_len)
        self.deduped_counts.append(deduped_len)
        
        chunk_metrics = {
            'chunk_index': chunk_index,
            'chunk_original_count': original_len,
            'chunk_deduplicated_count': deduped_len,
            'chunk_duplicates_removed': original_len - deduped_len,
            'chunk_compression_ratio': deduped_len / original_len if original_len else 0,
            **runtime_mem_metrics
        }
        
        wandb.log(chunk_metrics)

    def log_dedup_metrics(self, metrics: Dict[str, Any]):
       
        # Store metrics for final summary
        for key in self.dedup_metrics:
            if key in metrics:
                self.dedup_metrics[key].append(metrics[key])
        
        # Log standard metrics to wandb
        standard_metrics = {
            k: metrics[k] for k in [
                'processed_total',
                'chunk_kept',
                'chunk_duplicates',
                'global_unique',
                'global_duplicates',
                'duplicate_ratio'
            ] if k in metrics
        }
        wandb.log(standard_metrics)
        
        # Handle special cases
        if 'top_duplicates' in metrics:
            self._log_duplicate_examples(metrics['top_duplicates'])
        if 'similarity_scores' in metrics:
            self._log_similarity_distribution(metrics['similarity_scores'])

    def _log_duplicate_examples(self, top_duplicates: List[tuple[int, str]]):
        """Log examples of most frequent duplicates."""
        table = wandb.Table(
            columns=["Count", "Text Sample"],
            data=[[count, text[:200] + "..." if len(text) > 200 else text]
                 for count, text in top_duplicates]
        )
        wandb.log({"top_duplicates": table})

    def log_similarity_scores(self, similarities, stage="before"):
        if stage == "before":
            self.similarity_scores_before.extend(similarities)
        elif stage == "after":
            self.similarity_scores_after.extend(similarities)

    def _plot_similarity_histogram(self, scores, title):
        if not scores:
            return
        plt.figure(figsize=(8, 5))
        sns.histplot(scores, bins=20, kde=True)
        plt.title(title)
        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        wandb.log({title: wandb.Image(plt)})
        plt.close()

    def _plot_compression_line(self):
        if not self.chunk_indices:
            return
        plt.figure(figsize=(10, 5))
        plt.plot(self.chunk_indices, self.original_counts, label="Original")
        plt.plot(self.chunk_indices, self.deduped_counts, label="Deduplicated")
        plt.xlabel("Chunk Index")
        plt.ylabel("Document Count")
        plt.title("Compression Trend Across Chunks")
        plt.legend()
        wandb.log({"Compression Trend": wandb.Image(plt)})
        plt.close()

    def _plot_pie_chart(self):
        """Plot overall deduplication breakdown."""
        total_original = sum(self.original_counts)
        total_deduped = sum(self.deduped_counts)
        
        plt.figure(figsize=(5, 5))
        plt.pie(
            [total_deduped, total_original - total_deduped],
            labels=["Unique", "Duplicates"],
            autopct="%1.1f%%",
            startangle=90
        )
        plt.title("Overall Deduplication Breakdown")
        wandb.log({"deduplication_breakdown": wandb.Image(plt)})
        plt.close()
    
    def log_final_summary(self):
        """Generate and log final summary metrics and visualizations."""
        # Calculate totals

        self._plot_compression_line()
        self._plot_pie_chart()
        self._plot_similarity_histogram(
            self.similarity_scores_before, "Similarity Before Deduplication"
        )
        self._plot_similarity_histogram(
            self.similarity_scores_after, "Similarity After Deduplication"
        )

        total_original = sum(self.original_counts)
        total_deduped = sum(self.deduped_counts)
        
        # Final metrics
        final_metrics = {
            'total_original_count': total_original,
            'total_deduplicated_count': total_deduped,
            'total_duplicates_removed': total_original - total_deduped,
            'total_compression_ratio': total_deduped / total_original if total_original else 0,
            'avg_duplicate_ratio': sum(self.dedup_metrics['duplicate_ratio']) / 
                                 len(self.dedup_metrics['duplicate_ratio']) if self.dedup_metrics['duplicate_ratio'] else 0,
            'total_memory_used_mb': self.total_memory_mb,
            'total_algorithm_runtime_sec': self.total_runtime,
        }
        wandb.log(final_metrics)
        

    