from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from visionflow.core.inference.base import InferenceModelBase


@dataclass
class OcrResult:
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


class OcrModelBase(InferenceModelBase):
    @abstractmethod
    def extract(self, img: np.ndarray) -> List[OcrResult]:
        pass