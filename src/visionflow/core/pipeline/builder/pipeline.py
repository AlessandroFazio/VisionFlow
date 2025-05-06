# visionflow/core/pipeline/dsl.py

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type

import cv2

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.factory import EntityFactory
from visionflow.core.entity.iterator.hierarchy import EntityHierarchyIterator
from visionflow.core.entity.registry.base import EntityRegistryBase
from visionflow.core.entity.registry.registries import GraphEntityRegistry
from visionflow.core.entity.registry.visitors import EntityRegistryResolver
from visionflow.core.entity.utils import Entity
from visionflow.core.inference.base import InferenceServiceBase
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase
from visionflow.core.pipeline.steps.inference.classify import ClassifyStep
from visionflow.core.pipeline.steps.entity.build_entity import BuildEntityStep
from visionflow.core.pipeline.steps.entity.resolve_entity import ResolveEntityStep
from visionflow.core.pipeline.steps.functional.split_by import SplitByStep
from visionflow.core.pipeline.steps.inference.ocr import OcrStep
from visionflow.core.pipeline.steps.inference.detect import DetectStep
from visionflow.core.pipeline.steps.inference.filter import FilterStep
from visionflow.core.pipeline.steps.functional.process import ProcessStep
from visionflow.core.pipeline.pipeline import Pipeline
from visionflow.core.pipeline.steps.inference.ocr_regex_match import OcrRegexMatchStep
from visionflow.core.pipeline.steps.transforms.binarize import BinarizeStep
from visionflow.core.pipeline.steps.transforms.mask import MaskStep
from visionflow.core.pipeline.steps.transforms.crop import CropStep
from visionflow.core.pipeline.steps.transforms.resize import ResizeStep
from visionflow.core.pipeline.utils.matchers import BranchMatcherBase, DetectionClassMatcher
from visionflow.core.pipeline.utils.splitters import DetectionExchangeSplitter, ExchangeSplitterBase
from visionflow.core.pipeline.utils.providers import DetectionCoordinatesProvider, StaticCoordinatesProvider, TextProviderBase
from visionflow.core.regex.base import RegexMatcherBase
from visionflow.core.types import XyXyType

if TYPE_CHECKING:
    from visionflow.core.pipeline.builder.groups import StepGroup


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
        self._entity_registry: "EntityRegistryBase" = (
            self._parent._entity_registry 
            if parent else GraphEntityRegistry()
        )
        self._bound_entities: set[str] = (
            parent._bound_entities 
            if parent else set()
        )

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
    
    def static_mask(self, xyxy: XyXyType) -> "PipelineBuilder":
        return self.then(MaskStep(StaticCoordinatesProvider(xyxy)))
    
    def ocr_regex_match(self, matcher: RegexMatcherBase, provider: TextProviderBase) -> "PipelineBuilder":
        return self.then(OcrRegexMatchStep(matcher, provider))
    
    def _split_by(self) -> "SplitBuilder":
        self._split_builder = SplitBuilder(self)
        return self._split_builder
    
    def _end_branch(self) -> "SplitBuilder":
        return self._split_builder
    
    def end_class(self) -> "DetectionSplitBuilder":
        return self._end_branch()
    
    def split_by_detections(self) -> "DetectionSplitBuilder":
        self._split_builder = DetectionSplitBuilder(parent=self)
        return self._split_builder

    def process(self, fn: Callable[[PipelineContext, Exchange], Exchange]) -> "PipelineBuilder":
        return self.then(ProcessStep(fn))
    
    def build_entity(self, *classes: Type[EntityBase]) -> "PipelineBuilder":
        if not classes:
            raise ValueError("At least one entity class is required")

        for cls in classes:
            self._entity_registry.register_class(cls)

        for cls in classes:
            for cls_in_h in EntityHierarchyIterator(cls):
                name = Entity.name(cls_in_h)
                if name in self._bound_entities:
                    continue
                self._bound_entities.add(name)
                self.then(BuildEntityStep(EntityFactory(cls_in_h)))
        
        return self
    
    def apply(self, group: "StepGroup") -> "PipelineBuilder":
        return group.apply(self)
    
    def _build_context(self) -> PipelineContext:
        context = PipelineContext()
        if self._entity_registry:
            context.put(EntityRegistryBase.pipeline_ctx_key(), self._entity_registry)
            
        return context

    def build(self) -> Pipeline:
        if not self._parent:
            self.then(ResolveEntityStep(EntityRegistryResolver.default()))

        pipeline = Pipeline(self.name, self._steps, self._build_context())
        validation = pipeline.validate()
        if not validation.success():
            raise ValueError("")
        
        return pipeline


class SplitBuilder:
    def __init__(
        self, 
        parent: PipelineBuilder, 
        matcher: BranchMatcherBase, 
        exchange_splitter: ExchangeSplitterBase
    ) -> None:
        self._parent = parent
        self._matcher = matcher
        self._exchange_splitter = exchange_splitter
        self._branch_builders: Dict[str, PipelineBuilder] = {}
    
    def _branch(self, name: str) -> PipelineBuilder:
        builder = PipelineBuilder(name, self._parent.services, parent=self._parent)
        self._branch_builders[name] = builder
        return builder

    def end_split(self) -> PipelineBuilder:
        branches = {name: bb.build() for name, bb in self._branch_builders.items()}
        return self._parent.then(SplitByStep(branches))
    

class DetectionSplitBuilder(SplitBuilder):
    def __init__(self, parent: PipelineBuilder) -> None:
        super().__init__(
            parent=parent, 
            matcher=DetectionClassMatcher(), 
            exchange_splitter=DetectionExchangeSplitter()
        )

    def for_class(self, detection_class: str) -> "PipelineBuilder":
        return self._branch(detection_class)
