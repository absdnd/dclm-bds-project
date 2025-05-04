from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr

from .constants import DEDUPLICATION_METHODS


class DedupConfig(BaseSettings):
    # Dataset configuration
    dataset_name: str = Field(description="Hugging Face dataset name, e.g., 'c4'")
    dataset_config: str = Field(description="Dataset config, e.g., 'en.noclean'")
    dataset_split: str = Field(
        default="train", description="Split to use, e.g., 'train'"
    )
    # dataset_snapshot: str = Field(
    #     default="None", description="Snapshot to use"
    # )
    # dataset_language: str = Field(
    #     default="None", description="Language to use in case of multiple, e.g., 'en'"
    # )

    text_column: str = Field(default="text", description="Column with textual content")

    # General deduplication settings
    method: DEDUPLICATION_METHODS = Field(description="Deduplication method name")
    chunk_size: int = Field(
        default=1000, description="Number of samples to stream per chunk"
    )
    max_chunks: int | None = Field(
        default=None, description="Limit number of chunks to process"
    )

    # MinHash-specific settings (only used when method=minhash)
    minhash_threshold: float = Field(
        default=0.8,
        description="Similarity threshold for MinHash (0.0-1.0)",
        gt=0.0,
        lt=1.0,
    )
    minhash_num_hashes: int = Field(
        default=150, description="Number of hashes for MinHash", gt=0
    )
    minhash_debug_interval: int = Field(
        default=1000, description="How often to print debug info for MinHash", gt=0
    )

    # exact-specific settings (only used when method=exact)
    bloom_error_rate: float = Field(
        description="Error rate for bloom",
        gt=0.0,
        lt=1.0,
    )
    bloom_capacity: int = Field(description="Capacity for bloom", gt=0)
    bloom_debug_interval: int = Field(
        default=1000, description="How often to print debug info for bloom", gt=0
    )

    # Weights & Biases configuration
    wandb_project: str = Field(description="W&B project name")
    wandb_api_key: SecretStr | None = Field(
        default=None, description="W&B API Key (optional)"
    )

    # Hugging Face configuration
    hf_repo_id: str = Field(description="HF dataset repo, e.g. 'username/deduped-c4'")
    hf_private: bool = Field(
        default=True, description="Push HF repo as private (default True)"
    )

    hf_token: SecretStr | None = Field(
        default=None, description="Hugging Face API Token (optional)"
    )
    num_process: int | None = Field(default=None)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
