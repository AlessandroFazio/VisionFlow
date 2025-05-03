# visionflow/core/pipeline/dsl.py

from typing import Callable, Dict, List, Optional, Tuple

import cv2

from visionflow.core.inference.base import InferenceServiceBase
from visionflow.core.pipeline.base import Exchange, StepBase
from visionflow.core.pipeline.steps.inference.classify import ClassifyStep
from visionflow.core.pipeline.steps.misc.split_by import SplitByStep
from visionflow.core.pipeline.steps.inference.ocr import OcrStep
from visionflow.core.pipeline.steps.inference.detect import DetectStep
from visionflow.core.pipeline.steps.inference.filter import FilterStep
from visionflow.core.pipeline.steps.misc.process import ProcessStep
from visionflow.core.pipeline.steps.pipeline import Pipeline
from visionflow.core.pipeline.steps.transforms.binarize import BinarizeStep
from visionflow.core.pipeline.steps.transforms.mask import MaskStep
from visionflow.core.pipeline.steps.transforms.crop import CropStep
from visionflow.core.pipeline.steps.transforms.resize import ResizeStep
from visionflow.core.pipeline.utils.matchers import BranchMatcherBase, DetectionClassMatcher
from visionflow.core.pipeline.utils.multiplexers import DetectionMultiplexer, ExchangeMultiplexerBase
from visionflow.core.pipeline.utils.providers import DetectionCoordinatesProvider, StaticCoordinatesProvider
from visionflow.core.types import XyXyType


class PipelineBuilder:
    def __init__(
        self,
        name: str,
        services: Dict[str, InferenceServiceBase],
        parent: Optional["PipelineBuilder"] = None
    ):
        self.name = name
        self.services = services
        self._parent = parent
        self._steps: List[StepBase] = []
        self._split_builder: "SplitBuilder" = None

    def then(self, step: StepBase) -> "PipelineBuilder":
        self._steps.append(step)
        return self
    
    def binarize(self, normalize: bool=False) -> "PipelineBuilder":
        return self.then(BinarizeStep(normalize))
    
    def resize(self, reshape: Tuple[int, int], interpolation: int=cv2.INTER_CUBIC) -> "PipelineBuilder":
        return self.then(ResizeStep(reshape, interpolation))

    def detect(self, model_ref: str) -> "PipelineBuilder":
        svc = self.services[model_ref]
        return self.then(DetectStep(svc))
    
    def ocr(self, model_ref: str, ) -> "PipelineBuilder":
        svc = self.services[model_ref]
        return self.then(OcrStep(svc))

    def classify(self, model_ref: str, ) -> "PipelineBuilder":
        svc = self.services[model_ref]
        return self.then(ClassifyStep(svc))
    
    def filter(self, min_conf: float, ) -> "PipelineBuilder":
        return self.then(FilterStep(min_conf))
    
    def static_crop(self, xyxy: XyXyType) -> "PipelineBuilder":
        return self.then(CropStep(StaticCoordinatesProvider(xyxy)))
    
    def crop_to_detection(self) -> "PipelineBuilder":
        return self.then(CropStep(DetectionCoordinatesProvider()))
    
    def mask_detection(self) -> "PipelineBuilder":
        return self.then(MaskStep(DetectionCoordinatesProvider()))
    
    def split_by(self) -> "SplitBuilder":
        self._split_builder = SplitBuilder(self)
        return self._split_builder
    
    def end_branch(self) -> "SplitBuilder":
        return self._split_builder
    
    def split_by_detections(self) -> "SplitBuilder":
        self._split_builder = (
            SplitBuilder(self)
                .multiplexer(DetectionMultiplexer())
                .matcher(DetectionClassMatcher())
        )
        return self._split_builder

    def apply(self, fn: Callable[[Exchange], Exchange]) -> "PipelineBuilder":
        return self.then(ProcessStep(fn))

    def build(self) -> Pipeline:
        pipeline = Pipeline(self.name, self._steps)
        validation = pipeline.validate()
        if not validation.ok:
            raise ValueError("")
        return pipeline


class SplitBuilder:
    def __init__(self, parent: PipelineBuilder) -> None:
        self.parent = parent
        self._matcher = None
        self._multiplexer = None
        self._branch_builders: Dict[str, PipelineBuilder] = {}
    
    def matcher(self, matcher: BranchMatcherBase) -> "SplitBuilder":
        self._matcher = matcher
        return self
    
    def multiplexer(self, multiplexer: ExchangeMultiplexerBase) -> "SplitBuilder":
        self._multiplexer = multiplexer
        return self
    
    def branch(self, label: str) -> PipelineBuilder:
        builder = PipelineBuilder(label, self.parent.services, parent=self.parent)
        self._branch_builders[label] = builder
        return builder

    def end_split(self) -> PipelineBuilder:
        branches = {name: bb.build() for name, bb in self._branch_builders.items()}
        return self.parent.then(SplitByStep(branches))