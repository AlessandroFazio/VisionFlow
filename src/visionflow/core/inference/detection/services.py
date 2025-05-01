from typing import List
import numpy as np

import supervision as sv

from visionflow.core.inference.detection.base import DetectionResult, DetectionServiceBase
from visionflow.core.inference.mixins.roboflow import RoboflowInferenceMixin


class RoboflowDetectionService(RoboflowInferenceMixin, DetectionServiceBase):
    def detect(self, img: np.ndarray) -> List[DetectionResult]:
        results = self._model.infer(img)[0]
        detections = sv.Detections.from_inference(results)
        return DetectionResult.from_supervision(detections)