# visionflow/core/pipeline/steps/detect.py
import dataclasses
from visionflow.core.inference.detection.base import DetectionServiceBase
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class DetectStep(StepBase):
    def __init__(self, service: DetectionServiceBase) -> None:
        super().__init__()
        self.service = service

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        results = self.service.detect(exchange.image)
        return dataclasses.replace(exchange, detections=results)
