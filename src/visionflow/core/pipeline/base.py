# visionflow/core/pipeline/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Tuple, Type, TypeVar, Union
import uuid

import numpy as np
from prefect import Flow, Task, task
from prefect.tasks import task_input_hash

from visionflow.core.inference.classification.base import ClassificationResult
from visionflow.core.inference.detection.base import DetectionResult
from visionflow.core.inference.ocr.base import OcrResult


T = TypeVar('T')

@dataclass
class Exchange:
    execution_id: str
    image: np.ndarray
    original_image_shape: Tuple[int, int]
    detections: List[DetectionResult] = field(default_factory=list)
    classifications: List[ClassificationResult] = field(default_factory=list)
    ocr_results: List[OcrResult] = field(default_factory=list)
    private: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    ok: bool
    step: "StepBase"
    violations: List[str] = field(default_factory=list)

    def success(self) -> bool:
        return self.ok

    @classmethod
    def with_error(cls, *errors: str, step: "StepBase") -> "ValidationResult":
        return cls(ok=False, step=step, violations=list(errors))


class PipelineContext:
    def __init__(self) -> None:
        self._ctx_dict = {}

    def put(self, key: str, value: Any) -> None:
        if key in self._ctx_dict:
            raise ValueError()
        self._ctx_dict[key] = value

    def get(self, key: str, expect_type: Type[T]) -> T:
        if key not in self._ctx_dict:
            raise ValueError("")
        value = self._ctx_dict[key]
        if not isinstance(value, expect_type):
            raise ValueError("")
        return value


class StepBase(ABC):
    def __init__(self, name: str, cache_mins: int=-1) -> None:
        self.name = name
        self.cache_mins = cache_mins

    @abstractmethod
    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        pass
    
    def _execution_id(self) -> str:
        return f"{self.name}-{str(uuid.uuid4())}"

    def validate(self) -> ValidationResult:
        return ValidationResult(ok=True, step=self)

    def to_prefect(self) -> Union[Flow | Task]:
        task_deco = task
        if self.cache_mins > 0:
            task_deco = lambda name: task(
                name=name,
                cache_key_fn=task_input_hash,
                cache_expiration=timedelta(minutes=self.cache_mins)
            )

        @task_deco(name=self.name)
        def step_task(context: PipelineContext, exchange: Exchange) -> Exchange:
            return self.process(context, exchange)
        return step_task