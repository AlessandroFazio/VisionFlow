# visionflow/core/pipeline/steps/ocr_extract.py

from visionflow.core.inference.ocr.base import OcrServiceBase
from visionflow.core.pipeline.base import Exchange, StepBase


class OcrStep(StepBase):
    def __init__(self, service: OcrServiceBase) -> None:
        super().__init__(name="ocr")
        self.service = service

    def process(self, exchange: Exchange) -> Exchange:
        exchange.ocr_results = self.service.extract(exchange.image)
        return exchange
