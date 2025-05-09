from abc import ABC, abstractmethod
from typing import List

from visionflow.core.pipeline.base import Exchange, StepBase


class StepSelectorBase(ABC):
    @abstractmethod
    def select(self, exchange: Exchange, steps: List[StepBase]) -> List[StepBase]:
        pass

class DetectionClassSeletector(StepSelectorBase):
    def select(self, exchange: Exchange, steps: List[StepBase]) -> List[StepBase]:
        class_names = [d.class_name for d in exchange.detections]
        return [s for s in steps if s.tags.get("detection_class") in class_names]