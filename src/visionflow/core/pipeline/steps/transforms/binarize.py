# visionflow/core/pipeline/steps/binarize.py
import cv2
import numpy as np
from visionflow.core.pipeline.base import Exchange, StepBase


class BinarizeStep(StepBase):
    def __init__(self, normalize: bool, in_key: str, out_key: str) -> None:
        self.normalize = normalize
        super().__init__(name="binarize", in_key=in_key, out_key=out_key)
    
    def _binarize(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary.astype(np.float32) / 255.0 if self.normalize else binary
    
    def process(self, exchange: Exchange) -> Exchange:
        image = exchange.images[self.in_key]
        exchange.images[self.out_key] = self._binarize(image)
        return exchange