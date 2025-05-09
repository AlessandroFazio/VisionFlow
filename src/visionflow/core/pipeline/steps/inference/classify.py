# visionflow/core/pipeline/steps/classify.py

import dataclasses
from typing import Any, Dict, Optional
from visionflow.core.inference.classification.base import ClassificationModelBase
from visionflow.core.pipeline.base import RuntimeOptions, StepBase, StepRunContext


class ClassifyStep(StepBase):
    def __init__(
        self, 
        service: ClassificationModelBase,
        runtime_options: Optional[RuntimeOptions]=None, 
        tags: Optional[Dict[str, Any]]=None
    ) -> None:
        super().__init__(runtime_options=runtime_options, tags=tags)
        self.service = service

    def process(self, context: StepRunContext) -> StepRunContext:
        results = self.service.classify(context.exchange.image)
        exchange = dataclasses.replace(context.exchange, classifications=results)
        return dataclasses.replace(context, exchange=exchange)