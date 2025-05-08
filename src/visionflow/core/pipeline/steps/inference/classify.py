# visionflow/core/pipeline/steps/classify.py

import dataclasses
from visionflow.core.inference.classification.base import ClassificationModelBase
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class ClassifyStep(StepBase):
    def __init__(self, service: ClassificationModelBase) -> None:
        super().__init__()
        self.service = service

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        results = self.service.classify(exchange.image)
        return dataclasses.replace(exchange, classifications=results)