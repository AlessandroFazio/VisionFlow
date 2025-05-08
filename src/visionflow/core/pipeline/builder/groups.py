# visionflow/core/pipeline/dsl.py

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type

import cv2

from visionflow.core.inference.classification.base import ClassificationModelBase
from visionflow.core.inference.ocr.base import OcrModelBase
from visionflow.core.pipeline.utils.providers import OcrTextProvider
from visionflow.core.regex_matcher.base import RegexMatcherBase
from visionflow.core.types import XyXyType
from visionflow.core.pipeline.builder.pipeline import PipelineBuilder


class StepGroup(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def apply(self, builder: "PipelineBuilder") -> "PipelineBuilder":
        pass


class FilterCrop(StepGroup):
    def __init__(self, min_conf: float, xyxy: Optional[XyXyType]=None) -> None:
        super().__init__()
        self._min_conf = min_conf
        self._xyxy = xyxy

    def apply(self, builder: "PipelineBuilder") -> "PipelineBuilder":
        builder = builder.filter(self._min_conf)
        return (
            builder.crop_to_detection()
            if self._xyxy is not None
            else builder.static_crop(self._xyxy)
        )

    
class FilterMask(StepGroup):
    def __init__(self, min_conf: float, xyxy: Optional[XyXyType]=None) -> None:
        super().__init__()
        self._min_conf = min_conf
        self._xyxy = xyxy

    def apply(self, builder: "PipelineBuilder") -> "PipelineBuilder":
        builder = builder.filter(self._min_conf)
        return (
            builder.mask_detection()
            if self._xyxy is not None
            else builder.static_mask(self._xyxy)
        )


class FilterCropClassify(FilterCrop):
    def __init__(self, classifier: ClassificationModelBase, min_conf: float, xyxy: Optional[XyXyType]=None) -> None:
        super().__init__(min_conf=min_conf, xyxy=xyxy)
        self.classifier = classifier

    def apply(self, builder: "PipelineBuilder") -> "PipelineBuilder":
        return super().apply(builder).classify(self.classifier)


class FilterTransform(StepGroup):
    def __init__(
        self,
        base_cls: Type[StepGroup],
        min_conf: float,
        xyxy: Optional[XyXyType] = None,
        reshape: Optional[Tuple[int, int]] = None,
        interpolation: int = cv2.INTER_CUBIC,
        binarize: bool = True,
        normalize_binary: bool = False,
    ) -> None:
        self._base = base_cls(min_conf=min_conf, xyxy=xyxy)
        self._reshape = reshape
        self._interpolation = interpolation
        self._binarize = binarize
        self._normalize_binary = normalize_binary

    def apply(self, builder: "PipelineBuilder") -> "PipelineBuilder":
        builder = self._base.apply(builder)
        if self._reshape is not None:
            builder = builder.resize(self._reshape, self._interpolation)
        if self._binarize:
            builder = builder.binarize(self._normalize_binary)
        return builder

        
class OcrRegexMatcher(StepGroup):
    def __init__(self, ocr_model: OcrModelBase, matcher: RegexMatcherBase) -> None:
        super().__init__()
        self.ocr_model = ocr_model
        self.matcher = matcher
        
    def apply(self, builder: PipelineBuilder) -> PipelineBuilder:
        return (
            builder
                .ocr(self.ocr_model)
                .ocr_regex_match(self.matcher, OcrTextProvider())
        )