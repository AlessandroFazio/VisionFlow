# visionflow/core/image_source/base.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Iterator


class ImageSourceBase(ABC):
    @abstractmethod
    def images(self) -> Iterator[np.ndarray]:
        """Yield one or more images as NumPy arrays"""
        pass
