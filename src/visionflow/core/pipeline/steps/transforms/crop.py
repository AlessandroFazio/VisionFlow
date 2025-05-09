# visionflow/core/pipeline/steps/crop_image.py
import dataclasses
from typing import Any, Dict, Optional
from visionflow.core.pipeline.base import RuntimeOptions, StepBase, StepRunContext
from visionflow.core.pipeline.utils.providers import CoordinatesProviderBase


class CropStep(StepBase):
    def __init__(
        self, 
        provider: CoordinatesProviderBase,
        runtime_options: Optional[RuntimeOptions]=None, 
        tags: Optional[Dict[str, Any]]=None
    ) -> None:
        super().__init__(runtime_options=runtime_options, tags=tags)
        self.provider = provider

    def process(self, context: StepRunContext) -> StepRunContext:
        x1, y1, x2, y2 = self.provider(context.exchange)
        exchange = dataclasses.replace(context.exchange, context.exchange.image[y1:y2, x1:x2])
        return dataclasses.replace(context, exchange=exchange)

