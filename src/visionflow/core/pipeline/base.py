# visionflow/core/pipeline/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from multiprocessing import RLock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from visionflow.core.entity.registry.base import EntityRegistryBase
from visionflow.core.inference.classification.base import ClassificationResult
from visionflow.core.inference.detection.base import DetectionResult
from visionflow.core.inference.ocr.base import OcrResult
from visionflow.core.pipeline.backend.base import StepDispatcherBase
from visionflow.core.regex_matcher.matchers import RegexMatchResult


@dataclass(frozen=True)
class Exchange:
    image: np.ndarray
    original_image_shape: Tuple[int, int]
    detections: List[DetectionResult] = field(default_factory=list)
    classifications: List[ClassificationResult] = field(default_factory=list)
    ocr_results: List[OcrResult] = field(default_factory=list)
    ocr_regex_matches: List[RegexMatchResult] = field(default_factory=list)
    
    @classmethod
    def from_image(cls, img: np.ndarray) -> "Exchange":
        img_h, img_w = img.shape[:2]
        return cls(image=img, original_image_shape=(img_w, img_h))


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    step: "StepBase"
    violations: List[str] = field(default_factory=list)

    def success(self) -> bool:
        return self.ok

    @classmethod
    def with_error(cls, *errors: str, step: "StepBase") -> "ValidationResult":
        return cls(ok=False, step=step, violations=list(errors))
        
        
@dataclass(frozen=True)
class StepRunContext:
    entity_registry: EntityRegistryBase
    dispatcher: StepDispatcherBase
    exchange: Exchange
    _private = field(init=False, default_factory=dict)
    _lock = field(init=False, default_factory=RLock)
    
    def put_private(self, key: str, value: Any) -> None:
        with self._lock:
            self._private[key] = value

    def get_private(self, key: str) -> Any:
        with self._lock:
            return self._private[key]


@dataclass
class RuntimeOptions:
    cache_expiration: Optional[timedelta] = None
    retries: Optional[int] = None
    retry_delay_seconds: Optional[int] = None
    concurrency_limit: Optional[str] = None


class StepBase(ABC):
    def __init__(
        self,
        name: Optional[str]=None, 
        runtime_options: Optional[RuntimeOptions]=None, 
        tags: Dict[str, Any]=None
    ) -> None:
        self.name = name or self.__class__.__name__
        self.runtime_options = runtime_options or RuntimeOptions()
        self.tags = tags or {}

    @abstractmethod
    def process(self, context: StepRunContext) -> StepRunContext:
        pass

    def validate(self) -> ValidationResult:
        return ValidationResult(ok=True, step=self)
    
    def explain(self, depth: int = 0) -> str:
        return f"{'  ' * depth}- {self.name}"
    
    
class CompositeStep(StepBase):
    @abstractmethod
    def steps(self) -> List[StepBase]:
        pass
    
    @abstractmethod
    def rebuild_with_steps(self, steps: List[StepBase]) -> "CompositeStep":
        """Return a new instance with the given steps substituted in."""
        pass

    def validate(self) -> ValidationResult:
        validations = (step.validate() for step in self.steps())
        failures = [v for v in validations if not v.ok]
        return ValidationResult(ok=bool(len(failures)), step=self)