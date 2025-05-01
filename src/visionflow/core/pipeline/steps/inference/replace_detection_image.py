import numpy as np
from visionflow.core.inference.detection.base import DetectionResult
from visionflow.core.pipeline.base import Exchange, StepBase


class ReplaceDetectionImageStep(StepBase):
    def __init__(self, in_key: str, out_key: str, exclude_detection: bool=False) -> None:
        self.exclude_detection = exclude_detection
        super().__init__(name="crop_detection", in_key=in_key, out_key=out_key)

    def _replace(self, detection: DetectionResult, img: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = detection.xyxy
        if self.exclude_detection:
            img[y1:y2, x1:x2] = 0
            return img
        return img[y1:y2, x1:x2]
        

    def process(self, exchange: Exchange) -> Exchange:
        img = exchange.images[self.in_key]
        detections = exchange.detections[self.in_key]
        num_dets = len(detections)
        if num_dets == 0 or num_dets > 1:
            raise ValueError("")
        exchange.images[self.out_key] = self._replace(detections[0], img=img)
        return exchange