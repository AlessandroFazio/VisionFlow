from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from visionflow.core.pipeline.base import Exchange
from visionflow.core.types import XyXyType

T = TypeVar('T')


class ExchangeProviderBase(ABC, Generic[T]):
    @abstractmethod
    def get(self, exchange: Exchange) -> T:
        pass
    
    def __call__(self, exchange: Exchange) -> T:
        return self.get(exchange)

    
class CoordinatesProviderBase(ExchangeProviderBase[XyXyType]):
    pass


class DetectionCoordinatesProvider(CoordinatesProviderBase):
    def get(self, exchange: Exchange) -> XyXyType:
        return exchange.detections[0].xyxy


class StaticCoordinatesProvider(CoordinatesProviderBase):
    def __init__(self, xyxy: XyXyType) -> None:
        self.xyxy = xyxy

    def get(self, _: Exchange) -> XyXyType:
        return self.xyxy
    
    
class TextProviderBase(ExchangeProviderBase[str]):
    pass

 
class OcrTextProvider(TextProviderBase):
    def get(self, exchange: Exchange) -> str:
        last_lineno = 0
        text = ""
        for r in sorted(exchange.ocr_results, key=lambda r: (r.page, r.paragraph, r.line, r.word)):
            if r.line == last_lineno:
                text += f" {r.text}"
            else:
                text += f"\n{r.text}"
                last_lineno = r.line
        return text