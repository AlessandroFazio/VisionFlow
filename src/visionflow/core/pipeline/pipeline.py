import dataclasses
from typing import Callable, List, Union

import cv2
import numpy as np
from prefect import Flow, Task, flow
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase, ValidationResult


class Pipeline(StepBase):
    def __init__(self, name: str, steps: List[StepBase], context: PipelineContext) -> None:
        self.steps = steps
        self.context = context
        super().__init__(name=name)
    
    def _load_image(self, img_bytes: bytes) -> np.ndarray:
        img = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return img
    
    def _dispatch(self, context: PipelineContext, exchange: Exchange, executors: List[Callable[[PipelineContext, Exchange], Exchange]]) -> Exchange:
        for executor in executors:
            exchange = executor(context, exchange)
        return exchange

    def run(self, img_bytes: bytes) -> Exchange:
        img = self._load_image(img_bytes)
        img_h, img_w = img.shape[:2]
        exchange = Exchange(
            execution_id=self._execution_id(), 
            image=img, 
            original_image_shape=(img_w, img_h)
        )
        return self.process(self.context, exchange)
    
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