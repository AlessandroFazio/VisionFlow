# visionflow/core/pipeline/steps/filter_detections.py
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class FilterStep(StepBase):
    def __init__(self, min_confidence: float, category: str) -> None:
        super().__init__(name="filter")
        self.min_confidence = min_confidence
        self.category = category

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        items = [
            item for item in getattr(exchange, self.category, [])
            if item.confidence >= self.min_confidence
        ]
        setattr(exchange, self.category, items)
        return exchange