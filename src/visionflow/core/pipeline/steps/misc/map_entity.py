from typing import Type
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class MapEntityStep(StepBase):
    def __init__(self, cls: Type) -> None:
        self.cls = cls
        super().__init__(name="map_entity")

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        exchange.private["object"] = ...
        return exchange