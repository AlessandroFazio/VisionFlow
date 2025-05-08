from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from visionflow.core.inference.base import InferenceModelBase


@dataclass
class ClassificationResult:
    label: str
    confidence: float


class ClassificationModelBase(InferenceModelBase):
    @abstractmethod
    def classify(self, image: np.ndarray) -> List[ClassificationResult]:
        pass