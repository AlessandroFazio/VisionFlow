# visionflow/core/pipeline/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Dict, List, Union
import uuid

import cv2
import numpy as np
from prefect import Flow, Task, flow, task
from prefect.tasks import task_input_hash

from visionflow.core.inference.classification.base import ClassificationResult
from visionflow.core.inference.detection.base import DetectionResult
from visionflow.core.inference.ocr.base import OcrResult


@dataclass
class Exchange:
    execution_id: str
    image: np.ndarray
    detections: List[DetectionResult] = field(default_factory=list)
    classifications: List[ClassificationResult] = field(default_factory=list)
    ocr_results: List[OcrResult] = field(default_factory=list)
    private: Dict[str, Any] = field(default_factory=dict)
    children: Dict[str, "Exchange"] = field(default_factory=dict)

    def copy(self) -> "Exchange":
        return Exchange(
            execution_id=self.execution_id,
            image=self.image,
            detections=self.detections,
            classifications=self.classifications,
            ocr_results=self.ocr_results,
            private=self.private,
            children=self.children
        )


@dataclass
class ValidationResult:
    ok: bool
    step: "StepBase"
    violations: List[str] = field(default_factory=list)

    @classmethod
    def with_error(cls, *errors: str, step: "StepBase") -> "ValidationResult":
        return cls(ok=False, step=step, violations=list(errors))


class StepBase(ABC):
    def __init__(self, name: str, cache_mins: int=-1) -> None:
        self.name = name
        self.cache_mins = cache_mins

    @abstractmethod
    def process(self, exchange: Exchange) -> Exchange:
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
        def step_task(exchange: Exchange) -> None:
            return self.process(exchange)
        return step_task


class Pipeline(StepBase):
    def __init__(self, name: str, steps: List[StepBase]) -> None:
        super().__init__(name=name)
        self.steps = steps
    
    def _load_image(self, img_bytes: bytes) -> np.ndarray:
        img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return img
    
    def _dispatch(self, exchange: Exchange, executors: List[Callable[[Exchange], Exchange]]) -> Exchange:
        for executor in executors:
            exchange = executor(exchange.copy())
        return exchange

    def run(self, img_bytes: bytes) -> Exchange:
        img = self._load_image(img_bytes)
        return self.process(Exchange(self._execution_id(), image=img))
    
    def process(self, exchange: Exchange) -> Exchange:
        return self._dispatch(exchange, [step.process for step in self.steps])

    def to_prefect(self) -> Union[Task, Flow]:
        tasks = [step.to_prefect() for step in self.steps]
        
        @flow(name=self.name)
        def pipeline_flow(exchange: Exchange) -> Exchange:
            return self._dispatch(exchange, tasks)

        return pipeline_flow

    def validate(self) -> ValidationResult:
        validations = (step.validate() for step in self.steps)
        failures = [v for v in validations if not v.ok]
        return ValidationResult(ok=bool(len(failures)), step=self)
