from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from visionflow.core.inference.base import InferenceServiceBase


@dataclass
class ClassificationResult:
    label: str
    confidence: float


class ClassificationServiceBase(InferenceServiceBase):
    @abstractmethod
    def classify(self, image: np.ndarray) -> List[ClassificationResult]:
        pass