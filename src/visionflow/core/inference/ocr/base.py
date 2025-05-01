from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from visionflow.core.inference.base import InferenceServiceBase


@dataclass
class OcrDetection:
    page: int
    paragraph: int
    line: int
    word: int
    left: int
    top: int
    width: int
    height: int
    confidence: float
    text: str


@dataclass
class OcrResult:
    detections: List[OcrDetection]


class OcrServiceBase(InferenceServiceBase):
    @abstractmethod
    def extract(self, img: np.ndarray) -> OcrResult:
        pass