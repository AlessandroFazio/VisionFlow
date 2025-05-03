# visionflow/core/pipeline/steps/binarize.py
import cv2
import numpy as np
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class BinarizeStep(StepBase):
    def __init__(self, normalize: bool) -> None:
        super().__init__(name="binarize")
        self.normalize = normalize
    
    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange: 
        img = cv2.cvtColor(exchange.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        exchange.image = binary.astype(np.float32) / 255.0 if self.normalize else binary
        return exchange