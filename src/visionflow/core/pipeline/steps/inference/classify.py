# visionflow/core/pipeline/steps/classify.py
from typing import Dict, List
import numpy as np

from visionflow.core.inference.classification.base import ClassificationResult, ClassificationServiceBase
from visionflow.core.pipeline.base import Exchange, StepBase


class ClassifyStep(StepBase):
    def __init__(self, service: ClassificationServiceBase, in_key: str, out_key: str) -> None:
        self.service = service
        super().__init__(name="classify", in_key=in_key, out_key=out_key)

    def process(self, exchange: Exchange) -> Exchange:
        image = exchange.images[self.in_key]
        classifications = self.service.classify(image)
        exchange.classifications[self.out_key] = classifications
        return exchange