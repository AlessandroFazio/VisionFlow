# visionflow/core/pipeline/steps/detect.py
from visionflow.core.inference.detection.base import DetectionServiceBase
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class DetectStep(StepBase):
    def __init__(self, service: DetectionServiceBase) -> None:
        self.service = service
        super().__init__(name="detect")

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        exchange.detections = self.service.detect(exchange.image)
        return exchange
