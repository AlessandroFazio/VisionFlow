import dataclasses
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase
from visionflow.core.pipeline.utils.providers import CoordinatesProviderBase


class MaskStep(StepBase):
    def __init__(self, provider: CoordinatesProviderBase) -> None:
        super().__init__()
        self.provider = provider

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        x1, y1, x2, y2 = self.provider.get(exchange)
        img = exchange.image.copy()
        img[y1:y2, x1:x2] = 0
        return dataclasses.replace(exchange, image=img)
