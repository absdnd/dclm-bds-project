import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil


class MetricsLogger:
    def __init__(self):
        self.chunk_indices = []
        self.original_counts = []
        self.deduped_counts = []
        self.similarity_scores_before = []
        self.similarity_scores_after = []

        self.total_runtime = 0.0
        self.total_memory = 0.0

    def run_with_metrics(self, fn, *args, **kwargs):
        """Run a function and measure runtime and memory usage."""
        start_time = time.time()
        process = psutil.Process()
        start_mem = process.memory_info().rss

        result = fn(*args, **kwargs)

        end_time = time.time()
        end_mem = process.memory_info().rss
        runtime = end_time - start_time
        mem_used = (end_mem - start_mem) / 1024 / 1024  # in MB

        return result, {"runtime_sec": runtime, "memory_usage_mb": mem_used}

    def log_chunk_metrics(
        self, chunk_index, original_len, deduped_len, runtime_mem_metrics
    ):
        self.chunk_indices.append(chunk_index)
        self.original_counts.append(original_len)
        self.deduped_counts.append(deduped_len)

        chunk_metrics = {
            "chunk_index": chunk_index,
            "chunk_original_count": original_len,
            "chunk_deduplicated_count": deduped_len,
            "chunk_duplicates_removed": original_len - deduped_len,
            "chunk_compression_ratio": (
                deduped_len / original_len if original_len else 0
            ),
            "chunk_duplicates_removed_pct": (
                (1 - deduped_len / original_len) * 100 if original_len else 0
            ),
        }

        if runtime_mem_metrics:
            chunk_metrics.update(runtime_mem_metrics)

        wandb.log(chunk_metrics)

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
        total_original = sum(self.original_counts)
        total_deduped = sum(self.deduped_counts)
        values = [total_deduped, total_original - total_deduped]
        labels = ["Unique", "Duplicates"]
        plt.figure(figsize=(5, 5))
        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title("Overall Deduplication Breakdown")
        wandb.log({"Deduplication Breakdown": wandb.Image(plt)})
        plt.close()

    def log_final_summary(self):
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

        wandb.log(
            {
                "total_original_count": total_original,
                "total_deduplicated_count": total_deduped,
                "total_duplicates_removed": total_original - total_deduped,
                "total_compression_ratio": (
                    total_deduped / total_original if total_original else 0
                ),
                "total_duplicates_removed_pct": (
                    (1 - total_deduped / total_original) * 100 if total_original else 0
                ),
            }
        )
