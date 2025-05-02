from abc import ABC, abstractmethod
from typing import Dict, Tuple

from visionflow.core.pipeline.base import Exchange, StepBase


class BranchMatcherBase(ABC):
    @abstractmethod
    def match(self, exchange: Exchange, branches: Dict[str, StepBase]) -> Tuple[str, StepBase]:
        pass

class DetectionClassMatcher(BranchMatcherBase):
    def match(self, exchange: Exchange, branches: Dict[str, StepBase]) -> Tuple[str, StepBase]:
        class_name = exchange.detections[0].class_name
        for key, branch in branches.items():
            if key == class_name:
                return key, branch
        return None, None