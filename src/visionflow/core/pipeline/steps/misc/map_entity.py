from typing import Type
from visionflow.core.pipeline.base import Exchange, StepBase


class MapEntityStep(StepBase):
    def __init__(self, cls: Type) -> None:
        self.cls = cls
        super().__init__(name="map_entity")

    def process(self, exchange: Exchange) -> Exchange:
        exchange.private["object"] = ...
        return exchange