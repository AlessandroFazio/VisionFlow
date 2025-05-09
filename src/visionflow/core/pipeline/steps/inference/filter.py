# visionflow/core/pipeline/steps/filter_detections.py
import dataclasses
from typing import Any, Dict, Optional
from visionflow.core.pipeline.base import RuntimeOptions, StepBase, StepRunContext


class FilterStep(StepBase):
    def __init__(
        self, 
        min_confidence: float, 
        category: str,
        runtime_options: Optional[RuntimeOptions]=None, 
        tags: Optional[Dict[str, Any]]=None
    ) -> None:
        super().__init__(runtime_options=runtime_options, tags=tags)
        self.min_confidence = min_confidence
        self.category = category

    def process(self, context: StepRunContext) -> StepRunContext:
        items = [
            item for item in getattr(context.exchange, self.category, [])
            if item.confidence >= self.min_confidence
        ]
        exchange = dataclasses.replace(context.exchange, **{self.category: items})
        return dataclasses.replace(context, exchange=exchange)