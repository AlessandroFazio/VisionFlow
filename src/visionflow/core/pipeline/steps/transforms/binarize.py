# visionflow/core/pipeline/steps/binarize.py
import dataclasses
from typing import Any, Dict, Optional
import cv2
import numpy as np
from visionflow.core.pipeline.base import RuntimeOptions, StepBase, StepRunContext


class BinarizeStep(StepBase):
    def __init__(
        self, 
        normalize: bool,
        runtime_options: Optional[RuntimeOptions]=None, 
        tags: Optional[Dict[str, Any]]=None
    ) -> None:
        super().__init__(runtime_options=runtime_options, tags=tags)
        self.normalize = normalize
    
    def process(self, context: StepRunContext) -> StepRunContext: 
        img = cv2.cvtColor(context.exchange.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = binary.astype(np.float32) / 255.0 if self.normalize else binary
        exchange = dataclasses.replace(context.exchange, image=binary)
        return dataclasses.replace(context, exchange=exchange)