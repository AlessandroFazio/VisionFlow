# visionflow/core/pipeline/steps/crop_image.py
import dataclasses
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase
from visionflow.core.pipeline.utils.providers import CoordinatesProviderBase


class CropStep(StepBase):
    def __init__(self, provider: CoordinatesProviderBase) -> None:
        super().__init__(name="crop")
        self.provider = provider

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        x1, y1, x2, y2 = self.provider(exchange)
        return dataclasses.replace(exchange, exchange.image[y1:y2, x1:x2])


