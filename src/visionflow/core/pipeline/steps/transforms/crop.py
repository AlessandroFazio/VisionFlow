# visionflow/core/pipeline/steps/crop_image.py
import numpy as np
from visionflow.core.inference.detection.base import DetectionResult
from visionflow.core.pipeline.base import Exchange, StepBase


class DynamicCropStep(StepBase):
    def __init__(self, roi: str, in_key: str, out_key: str) -> None:
        self.roi = roi
        super().__init__(name="dynamic_crop", in_key=in_key, out_key=out_key)

    def _crop(self, detection: DetectionResult, img: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = detection.xyxy
        return img[y1:y2, x1:x2]

    def process(self, exchange: Exchange) -> Exchange:
        images = exchange.images[self.in_key]
        if self.roi == "detection_box":
            detections = exchange.detections[self.in_key]
            for detections in detections:
                cropped = [self._crop(detection, img) for detection in   ]
        return img[y1:y2, x1:x2]


