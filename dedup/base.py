from abc import ABC, abstractmethod


class Deduplicator(ABC):
    @abstractmethod
    def run(self, examples: list[dict]) -> list[dict]:
        """Returns a deduplicated HuggingFace dataset."""
        pass
