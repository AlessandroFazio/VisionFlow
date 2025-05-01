# visionflow/core/pipeline/steps/filter_detections.py
from visionflow.core.pipeline.base import Exchange, StepBase


class FilterStep(StepBase):
    def __init__(self, min_confidence: float, category: str, in_key: str, out_key: str) -> None:
        self.min_confidence = min_confidence
        self.category = category
        super().__init__(name="filter", in_key=in_key, out_key=out_key)

    def process(self, exchange: Exchange) -> Exchange:
        category_items = getattr(exchange, self.category, {})
        items = [
            item for item in category_items.get(self.in_key, [])
            if item.confidence >= self.min_confidence
        ]
        category_items[self.out_key] = items
        return exchange