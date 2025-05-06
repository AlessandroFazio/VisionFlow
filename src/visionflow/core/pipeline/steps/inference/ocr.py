# visionflow/core/pipeline/steps/ocr_extract.py

import dataclasses
from visionflow.core.inference.ocr.base import OcrServiceBase
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class OcrStep(StepBase):
    def __init__(self, service: OcrServiceBase) -> None:
        super().__init__()
        self.service = service

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        results = self.service.extract(exchange.image)
        return dataclasses.replace(exchange, ocr_results=results)
