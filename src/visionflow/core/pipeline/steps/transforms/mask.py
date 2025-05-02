from visionflow.core.pipeline.base import Exchange, StepBase
from visionflow.core.pipeline.utils.providers import CoordinatesProviderBase


class MaskStep(StepBase):
    def __init__(self, provider: CoordinatesProviderBase) -> None:
        super().__init__(name="mask")
        self.provider = provider

    def process(self, exchange: Exchange) -> Exchange:
        x1, y1, x2, y2 = self.provider.get(exchange)
        exchange.image[y1:y2, x1:x2]
        return exchange
