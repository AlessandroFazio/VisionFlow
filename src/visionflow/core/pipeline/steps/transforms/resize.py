import dataclasses
from typing import Any, Dict, Optional, Tuple
import cv2
from visionflow.core.pipeline.base import RuntimeOptions, StepBase, StepRunContext


class ResizeStep(StepBase):
    def __init__(
        self, 
        reshape: Tuple[int, int], 
        interpolation: int,
        runtime_options: Optional[RuntimeOptions]=None, 
        tags: Optional[Dict[str, Any]]=None
    ) -> None:
        super().__init__(runtime_options=runtime_options, tags=tags)
        self.fx, self.fy = reshape
        self.interpolation = interpolation
    
    def process(self, context: StepRunContext) -> StepRunContext:
        img = cv2.resize(
            context.exchange.image, (0,0), fx=self.fx, fy=self.fy, interpolation=self.interpolation
        )
        exchange = dataclasses.replace(context.exchange, image=img)
        return dataclasses.replace(context, exchange=exchange)