import itertools

from .minhash import MinHashDeduplicator
from .exact import ExactHashDeduplicator

from .registry import DEDUPLICATOR_REGISTRY
from .utils import push_chunk_to_hub, init_wandb
from .config import DedupConfig
from .metrics import MetricsLogger

from datasets import load_dataset

metrics_logger = MetricsLogger()


def chunked_iterable(iterable, chunk_size):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, chunk_size))
        if not chunk:
            break
        yield chunk


def run_pipeline(cfg: DedupConfig):

    init_wandb(cfg)

    print(f"Loading streaming dataset: {cfg.dataset_name}...")
    raw_stream = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.dataset_split,
        streaming=True,
        trust_remote_code=True,
    )

    dedup_cls = DEDUPLICATOR_REGISTRY.get(cfg.method)

    if dedup_cls is None:
        raise ValueError(
            f"Unknown deduplication method '{cfg.method}'. "
            f"Available methods: {list(DEDUPLICATOR_REGISTRY.keys())}"
        )

    deduper = dedup_cls(cfg = cfg)

    print("Processing in streaming chunks (global deduplication)...")
    
    for i, chunk in enumerate(chunked_iterable(raw_stream, cfg.chunk_size)):

        #deduped_chunk, perf = metrics_logger.run_with_metrics(deduper.run, chunk)

        (deduped_chunk, metrics), perf = metrics_logger.run_with_metrics(
            lambda: deduper.run(chunk)
        )

        print(f"Chunk {i + 1}: {len(chunk)} → {len(deduped_chunk)} after deduplication")

        metrics_logger.log_chunk_metrics(
            chunk_index=i + 1,
            original_len=len(chunk),
            deduped_len=len(deduped_chunk),
            runtime_mem_metrics=perf,
        )

        # Log method-specific metrics if available
        metrics_logger.log_dedup_metrics(metrics)

        push_chunk_to_hub(
            chunk=deduped_chunk,
            repo_id=cfg.hf_repo_id,
            split_name=f"chunk_{i+1}",
            private=cfg.hf_private,
            token=cfg.hf_token.get_secret_value() if cfg.hf_token else None,
            data_dir=f"deduplicated_{cfg.dataset_config}_using_{cfg.method}_with_{cfg.max_chunks*cfg.chunk_size//1000}k_data",
        )

        if cfg.max_chunks is not None and i + 1 >= cfg.max_chunks:
            break

    metrics_logger.log_final_summary()
    print("Streaming deduplication complete.")
