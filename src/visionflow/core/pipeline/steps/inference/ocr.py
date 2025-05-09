# visionflow/core/pipeline/steps/ocr_extract.py

import dataclasses
from typing import Any, Dict, Optional
from visionflow.core.inference.ocr.base import OcrModelBase
from visionflow.core.pipeline.base import RuntimeOptions, StepBase, StepRunContext


class OcrStep(StepBase):
    def __init__(
        self, 
        service: OcrModelBase,
        runtime_options: Optional[RuntimeOptions]=None, 
        tags: Optional[Dict[str, Any]]=None
    ) -> None:
        super().__init__(runtime_options=runtime_options, tags=tags)
        self.service = service

    def process(self, context: StepRunContext) -> StepRunContext:
        results = self.service.extract(context.exchange.image)
        exchange = dataclasses.replace(context.exchange, ocr_results=results)
        return dataclasses.replace(context, exchange=exchange)