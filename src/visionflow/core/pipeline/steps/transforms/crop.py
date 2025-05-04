# visionflow/core/pipeline/steps/crop_image.py
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase
from visionflow.core.common.coordinates.providers import CoordinatesStrategyBase


class CropStep(StepBase):
    def __init__(self, provider: CoordinatesStrategyBase) -> None:
        super().__init__(name="crop")
        self.provider = provider

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        x1, y1, x2, y2 = self.provider(exchange)
        exchange.image = exchange.image[y1:y2, x1:x2]
        return exchange


