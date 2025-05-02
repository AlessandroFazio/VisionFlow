from typing import Tuple
import cv2
from visionflow.core.pipeline.base import Exchange, StepBase


class ResizeStep(StepBase):
    def __init__(self, reshape: Tuple[int, int], interpolation: int) -> None:
        self.fx, self.fy = reshape
        self.interpolation = interpolation
        super().__init__(name="resize")
    
    def process(self, exchange: Exchange) -> Exchange:
        exchange.image = cv2.resize(
            exchange.image, (0,0), fx=self.fx, fy=self.fy, interpolation=self.interpolation
        )
        return exchange