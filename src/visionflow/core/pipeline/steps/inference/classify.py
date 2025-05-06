# visionflow/core/pipeline/steps/classify.py

import dataclasses
from visionflow.core.inference.classification.base import ClassificationServiceBase
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class ClassifyStep(StepBase):
    def __init__(self, service: ClassificationServiceBase) -> None:
        super().__init__(name="classify")
        self.service = service

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        results = self.service.classify(exchange.image)
        return dataclasses.replace(exchange, classifications=results)