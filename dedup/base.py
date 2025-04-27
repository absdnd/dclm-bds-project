from abc import ABC, abstractmethod


class Deduplicator(ABC):
    @abstractmethod
    def __init__(self, text_column: str) -> None:
        pass

    @abstractmethod
    def run(self, examples: list[dict]) -> list[dict]:
        """Returns a deduplicated HuggingFace dataset."""
        pass
