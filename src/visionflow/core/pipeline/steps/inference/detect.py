# visionflow/core/pipeline/steps/detect.py
import dataclasses
from typing import Any, Dict, Optional
from visionflow.core.inference.detection.base import DetectionModelBase
from visionflow.core.pipeline.base import RuntimeOptions, StepBase, StepRunContext


class DetectStep(StepBase):
    def __init__(
        self, 
        service: DetectionModelBase,
        runtime_options: Optional[RuntimeOptions]=None, 
        tags: Optional[Dict[str, Any]]=None
    ) -> None:
        super().__init__(runtime_options=runtime_options, tags=tags)
        self.service = service

    def process(self, context: StepRunContext) -> StepRunContext:
        results = self.service.detect(context.exchange.image)
        exchange = dataclasses.replace(context.exchange, detections=results)
        return dataclasses.replace(context, exchange=exchange)