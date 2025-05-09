# visionflow/core/pipeline/dsl.py

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type, Set

import cv2

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.factory import EntityFactory
from visionflow.core.entity.iterator.hierarchy import EntityHierarchyIterator
from visionflow.core.entity.registry.base import EntityRegistryVisitorBase
from visionflow.core.entity.registry.visitors import EntityRegistryResolver
from visionflow.core.entity.utils import Entity
from visionflow.core.inference.classification.base import ClassificationModelBase
from visionflow.core.inference.detection.base import DetectionModelBase
from visionflow.core.inference.ocr.base import OcrModelBase
from visionflow.core.pipeline.base import StepBase, StepRunContext
from visionflow.core.pipeline.steps.inference.classify import ClassifyStep
from visionflow.core.pipeline.steps.entity.build_entity import BuildEntityStep
from visionflow.core.pipeline.steps.entity.resolve_entity import ResolveEntityStep
from visionflow.core.pipeline.steps.utility.branch import BranchStep
from visionflow.core.pipeline.steps.inference.ocr import OcrStep
from visionflow.core.pipeline.steps.inference.detect import DetectStep
from visionflow.core.pipeline.steps.inference.filter import FilterStep
from visionflow.core.pipeline.steps.utility.process import ProcessStep
from visionflow.core.pipeline.pipeline import Pipeline
from visionflow.core.pipeline.steps.inference.ocr_regex_match import OcrRegexMatchStep
from visionflow.core.pipeline.steps.transforms.binarize import BinarizeStep
from visionflow.core.pipeline.steps.transforms.mask import MaskStep
from visionflow.core.pipeline.steps.transforms.crop import CropStep
from visionflow.core.pipeline.steps.transforms.resize import ResizeStep
from visionflow.core.pipeline.utils.selector import StepSelectorBase, DetectionClassSeletector
from visionflow.core.pipeline.utils.splitters import DetectionSplitter, ExchangeSplitterBase
from visionflow.core.pipeline.utils.providers import DetectionCoordinatesProvider, StaticCoordinatesProvider, TextProviderBase
from visionflow.core.regex_matcher.base import RegexMatcherBase
from visionflow.core.types import XyXyType

if TYPE_CHECKING:
    from visionflow.core.pipeline.builder.groups import StepGroup


class PipelineBuilder:
    def __init__(
        self,
        name: str,
        parent: Optional["PipelineBuilder"] = None
    ):
        self.name = name
        self._parent = parent
        self._tags = {}
        self._steps: List[StepBase] = []
        self._branch_builder: "BranchBuilder" = None
        self._init_or_inherit(parent)
        
    def _init_or_inherit(self, parent: "PipelineBuilder") -> None:
        if parent:
            self._bound_entities = parent._bound_entities
        else:
            self._bound_entities: Set[str] = set()

    def then(self, step: StepBase) -> "PipelineBuilder":
        self._steps.append(step)
        return self
    
    def with_tags(self, **kwargs) -> "PipelineBuilder":
        self._tags.update(**kwargs)
        return self
    
    def binarize(self, normalize: bool=False) -> "PipelineBuilder":
        return self.then(BinarizeStep(normalize))
    
    def resize(self, reshape: Tuple[int, int], interpolation: int=cv2.INTER_CUBIC) -> "PipelineBuilder":
        return self.then(ResizeStep(reshape, interpolation))

    def detect(self, model: DetectionModelBase) -> "PipelineBuilder":
        return self.then(DetectStep(model))
    
    def ocr(self, model: OcrModelBase) -> "PipelineBuilder":
        return self.then(OcrStep(model))

    def classify(self, model: ClassificationModelBase) -> "PipelineBuilder":
        return self.then(ClassifyStep(model))
    
    def filter(self, min_conf: float) -> "PipelineBuilder":
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
    
    def branches(self) -> "BranchBuilder":
        self._branch_builder = BranchBuilder(self)
        return self._branch_builder
    
    def end_branches(self) -> "BranchBuilder":
        return self._branch_builder
    
    def end_class(self) -> "DetectionBranchBuilder":
        return self.end_branches()
    
    def branch_detections(self) -> "DetectionBranchBuilder":
        self._branch_builder = DetectionBranchBuilder(parent=self)
        return self._branch_builder

    def process(self, fn: Callable[[StepRunContext], StepRunContext]) -> "PipelineBuilder":
        return self.then(ProcessStep(fn))
    
    def build_entities(self, *classes: Type[EntityBase]) -> "PipelineBuilder":
        if not classes:
            raise ValueError("At least one entity class is required")
        
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
    
    def resolve_entities(self, resolver: Optional[EntityRegistryVisitorBase]=None) -> "PipelineBuilder":
        resolver = resolver or EntityRegistryResolver.default()
        return self.then(ResolveEntityStep(resolver))
    
    def build(self) -> Pipeline:
        pipeline = Pipeline(self.name, self._steps)
        validation = pipeline.validate()
        if not validation.success():
            raise ValueError("")
        
        return pipeline


class BranchBuilder:
    def __init__(
        self, 
        parent: PipelineBuilder, 
        selector: StepSelectorBase, 
        splitter: ExchangeSplitterBase
    ) -> None:
        self._parent = parent
        self._selector = selector
        self._splitter = splitter
        self._branch_builders: Dict[str, PipelineBuilder] = {}
    
    def branch(self, name: str) -> PipelineBuilder:
        qualname = f"{self._parent.name}||{name}"
        builder = PipelineBuilder(name, parent=self._parent)
        self._branch_builders[name] = builder
        return builder

    def end_branch(self) -> PipelineBuilder:
        branches = {name: bb.build() for name, bb in self._branch_builders.items()}
        return self._parent.then(BranchStep(branches))
    

class DetectionBranchBuilder(BranchBuilder):
    def __init__(self, parent: PipelineBuilder) -> None:
        super().__init__(
            parent=parent, 
            selector=DetectionClassSeletector(), 
            splitter=DetectionSplitter()
        )

    def for_class(self, detection_class: str) -> "PipelineBuilder":
        return (
            self.branch(detection_class)
                .with_tags(detection_class=detection_class)
        )
