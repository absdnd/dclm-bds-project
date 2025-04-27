from pydantic_settings import BaseSettings
from pydantic import Field, SecretStr

from .constants import DEDUPLICATION_METHODS


class DedupConfig(BaseSettings):
    dataset_name: str = Field(description="Hugging Face dataset name, e.g., 'c5'")
    dataset_config: str = Field(description="Dataset config, e.g., 'en'")
    dataset_split: str = Field(
        default="train", description="Split to use, e.g., 'train'"
    )
    text_column: str = Field(default="text", description="Column with textual content")

    method: DEDUPLICATION_METHODS = Field(description="Deduplication method name")
    chunk_size: int = Field(
        default=1000, description="Number of samples to stream per chunk"
    )
    max_chunks: int | None = Field(description="Limit number of chunks to process")

    wandb_project: str = Field(description="W&B project name")
    wandb_api_key: SecretStr | None = Field(
        default=None, description="W&B API Key (optional)"
    )

    hf_repo_id: str = Field(description="HF dataset repo, e.g. 'username/deduped-c4'")
    hf_private: bool = Field(
        default=True, description="Push HF repo as private (default True)"
    )
    hf_token: SecretStr | None = Field(
        default=None, description="Hugging Face API Token (optional)"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
