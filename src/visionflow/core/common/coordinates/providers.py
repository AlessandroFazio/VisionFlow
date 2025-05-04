from abc import ABC, abstractmethod
from enum import Enum

from visionflow.core.pipeline.base import Exchange
from visionflow.core.common.types import XyXyType


class CoordinatesStrategyBase(ABC):
    @abstractmethod
    def get(self, exchange: Exchange) -> XyXyType:
        pass

    def __call__(self, exchange: Exchange) -> XyXyType:
        return self.get(exchange)


class DetectionCoordinatesProvider(CoordinatesStrategyBase):
    def get(self, exchange: Exchange) -> XyXyType:
        return exchange.detections[0].xyxy


class StaticCoordinatesProvider(CoordinatesStrategyBase):
    def __init__(self, xyxy: XyXyType) -> None:
        self.xyxy = xyxy

    def get(self, _: Exchange) -> XyXyType:
        return self.xyxy