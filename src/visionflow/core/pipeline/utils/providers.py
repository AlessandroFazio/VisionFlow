from abc import ABC, abstractmethod

from visionflow.core.pipeline.base import Exchange
from visionflow.core.types import XyXyType


class CoordinatesProviderBase(ABC):
    @abstractmethod
    def get(self, exchange: Exchange) -> XyXyType:
        pass


class DetectionCoordinatesProvider(CoordinatesProviderBase):
    def get(self, exchange: Exchange) -> XyXyType:
        return exchange.detections[0].xyxy


class StaticCoordinatesProvider(CoordinatesProviderBase):
    def __init__(self, xyxy: XyXyType) -> None:
        self.xyxy = xyxy

    def get(self, _: Exchange) -> XyXyType:
        return self.xyxy