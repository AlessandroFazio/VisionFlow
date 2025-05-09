from abc import ABC, abstractmethod
import dataclasses
from typing import List

from visionflow.core.pipeline.base import Exchange


class ExchangeSplitterBase(ABC):
    @abstractmethod
    def split(self, exchange: Exchange) -> List[Exchange]:
        pass

class DetectionSplitter(ExchangeSplitterBase):
    def split(self, exchange: Exchange) -> List[Exchange]:
        return [
            dataclasses.replace(exchange, detections=[det])
            for det in exchange.detections
        ]
        
        
class ZeroSplitter(ExchangeSplitterBase):
    def split(self, exchange: Exchange) -> List[Exchange]:
        return [exchange]