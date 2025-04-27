# utils/push_to_hf.py
from datasets import Dataset, DatasetDict


def push_chunk_to_hub(
    chunk,
    repo_id: str,
    data_dir: str,
    split_name: str,
    private: bool,
    token: str | None = None,
):
    """
    Push a single chunk as a dataset to the Hugging Face Hub.
    Uses DatasetDict to overwrite chunk by chunk.
    """
    ds = Dataset.from_list(chunk)
    ds_dict = DatasetDict({split_name: ds})

    ds_dict.push_to_hub(
        repo_id=repo_id,
        private=private,
        token=token,
        max_shard_size="500MB",
        data_dir=data_dir,
    )
