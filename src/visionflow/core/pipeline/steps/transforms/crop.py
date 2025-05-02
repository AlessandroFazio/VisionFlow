# visionflow/core/pipeline/steps/crop_image.py
from visionflow.core.pipeline.base import Exchange, StepBase
from visionflow.core.pipeline.utils.providers import CoordinatesProviderBase
from visionflow.core.types import XyXyType

    
class StaticCoordinatesProvider(CoordinatesProviderBase):
    def __init__(self, xyxy: XyXyType) -> None:
        self.xyxy = xyxy

    def get(self, _: Exchange) -> XyXyType:
        return self.xyxy

class CropStep(StepBase):
    def __init__(self, provider: CoordinatesProviderBase) -> None:
        super().__init__(name="crop")
        self.provider = provider

    def process(self, exchange: Exchange) -> Exchange:
        x1, y1, x2, y2 = self.provider.get(exchange)
        exchange.image = exchange.image[y1:y2, x1:x2]
        return exchange


