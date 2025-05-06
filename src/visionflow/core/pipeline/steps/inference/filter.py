# visionflow/core/pipeline/steps/filter_detections.py
import dataclasses
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class FilterStep(StepBase):
    def __init__(self, min_confidence: float, category: str) -> None:
        super().__init__()
        self.min_confidence = min_confidence
        self.category = category

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        items = [
            item for item in getattr(exchange, self.category, [])
            if item.confidence >= self.min_confidence
        ]
        return dataclasses.replace(exchange, **{self.category: items})