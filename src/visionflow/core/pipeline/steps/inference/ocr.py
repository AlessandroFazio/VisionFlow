# visionflow/core/pipeline/steps/ocr_extract.py

from visionflow.core.inference.ocr.base import OcrServiceBase
from visionflow.core.pipeline.base import Exchange, StepBase


class OcrStep(StepBase):
    def __init__(self, service: OcrServiceBase, in_key: str, out_key: str) -> None:
        self.service = service
        super().__init__(name="ocr", in_key=in_key, out_key=out_key)
        
    def process(self, exchange: Exchange) -> Exchange:
        images = exchange.images[self.in_key]
        results = [self.service.extract(img) for img in images]
        exchange.ocr_results[self.out_key] = results
        return exchange
