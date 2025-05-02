from abc import ABC, abstractmethod
from typing import List

from visionflow.core.pipeline.base import Exchange


class ExchangeMultiplexerBase(ABC):
        @abstractmethod
        def multiplex(self, exchange: Exchange) -> List[Exchange]:
            pass

class DetectionMultiplexer(ExchangeMultiplexerBase):
    def multiplex(self, exchange: Exchange) -> List[Exchange]:
        exchanges = []
        for det in exchange.detections:
            copy = exchange.copy()
            copy.detections = [det]
            exchanges.append(copy)
        return exchanges