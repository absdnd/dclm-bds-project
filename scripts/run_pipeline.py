from dedup.config import DedupConfig
from dedup.runner import run_pipeline

if __name__ == "__main__":
    cfg = DedupConfig()
    deduped = run_pipeline(cfg)
