# visionflow/core/pipeline/steps/detect.py
from visionflow.core.inference.detection.base import DetectionServiceBase
from visionflow.core.pipeline.base import Exchange, StepBase


class DetectStep(StepBase):
    def __init__(self, service: DetectionServiceBase, in_key: str, out_key: str) -> None:
        self.service = service
        super().__init__(name="detect", in_key=in_key, out_key=out_key)

    def process(self, exchange: Exchange) -> Exchange:
        image = exchange.images[self.in_key]
        detections = self.service.detect(image)
        exchange.detections[self.out_key] = detections
        return exchange
