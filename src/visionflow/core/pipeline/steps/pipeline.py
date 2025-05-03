import dataclasses
from typing import Callable, List, Union

import cv2
import numpy as np
from prefect import Flow, Task, flow
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase, ValidationResult


class Pipeline(StepBase):
    def __init__(self, name: str, steps: List[StepBase]) -> None:
        super().__init__(name=name)
        self.steps = steps
    
    def _load_image(self, img_bytes: bytes) -> np.ndarray:
        img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return img
    
    def _dispatch(self, context: PipelineContext, exchange: Exchange, executors: List[Callable[[PipelineContext, Exchange], Exchange]]) -> Exchange:
        for executor in executors:
            exchange = executor(context, dataclasses.replace(exchange))
        return exchange

    def run(self, context: PipelineContext, img_bytes: bytes) -> Exchange:
        img = self._load_image(img_bytes)
        return self.process(context, Exchange(self._execution_id(), image=img))
    
    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        return self._dispatch(context, exchange, [step.process for step in self.steps])

    def to_prefect(self) -> Union[Task, Flow]:
        tasks = [step.to_prefect() for step in self.steps]
        
        @flow(name=self.name)
        def pipeline_flow(context: PipelineContext, exchange: Exchange) -> Exchange:
            return self._dispatch(context, exchange, tasks)

        return pipeline_flow

    def validate(self) -> ValidationResult:
        validations = (step.validate() for step in self.steps)
        failures = [v for v in validations if not v.ok]
        return ValidationResult(ok=bool(len(failures)), step=self)
