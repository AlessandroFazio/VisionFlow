import dataclasses
from typing import Tuple
import cv2
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class ResizeStep(StepBase):
    def __init__(self, reshape: Tuple[int, int], interpolation: int) -> None:
        super().__init__()
        self.fx, self.fy = reshape
        self.interpolation = interpolation
    
    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        img = cv2.resize(
            exchange.image, (0,0), fx=self.fx, fy=self.fy, interpolation=self.interpolation
        )
        return dataclasses.replace(exchange, image=img)