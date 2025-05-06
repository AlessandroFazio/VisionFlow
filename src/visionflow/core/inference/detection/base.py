from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import supervision as sv
import numpy as np

from visionflow.core.inference.base import InferenceServiceBase
from visionflow.core.types import XyXyType




@dataclass
class DetectionResult:
    xyxy: XyXyType
    confidence: float
    class_name: str

    @classmethod
    def from_supervision(cls, detections: sv.Detections) -> List["DetectionResult"]:
        return [
            DetectionResult(
                class_name=str(data["class_name"]),
                xyxy=tuple(map(int, xyxy)),
                confidence=float(confidence)
            ) for xyxy, _, confidence, _, _, data in detections
        ]


class DetectionServiceBase(InferenceServiceBase):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        pass