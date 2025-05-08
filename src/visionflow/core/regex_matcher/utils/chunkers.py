
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple


class ChunkerStrategyBase(ABC):
    @abstractmethod
    def chunks(self, text: str) -> Iterable[Tuple[Optional[int], str]]:
        pass


class LineChunker(ChunkerStrategyBase):
    def chunks(self, text: str) -> Iterable[Tuple[int, str]]:
        return enumerate(text.splitlines())


class FullTextChunker(ChunkerStrategyBase):
    def chunks(self, text: str) -> Iterable[Tuple[None, str]]:
        yield None, text
