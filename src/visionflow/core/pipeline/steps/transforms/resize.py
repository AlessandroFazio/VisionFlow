from typing import Tuple
import cv2
from visionflow.core.pipeline.base import Exchange, StepBase


class ResizeStep(StepBase):
    def __init__(self, reshape: Tuple[int, int], interpolation: int, in_key: str, out_key: str) -> None:
        self.fx, self.fy = reshape
        self.interpolation = interpolation
        super().__init__(name="resize", in_key=in_key, out_key=out_key)
    
    def process(self, exchange: Exchange) -> Exchange:
        image = exchange.images[self.in_key]
        resized = cv2.resize(image, (0,0), fx=self.fx, fy=self.fy, interpolation=self.interpolation)
        exchange.images[self.out_key] = resized
        return exchange