# visionflow/core/pipeline/dsl.py

from typing import Dict, List, Optional, Tuple, TypeVar, Generic

import cv2

from visionflow.core.inference.base import InferenceServiceBase
from visionflow.core.pipeline.base import StepBase, Pipeline

from visionflow.core.pipeline.steps.inference.classify import ClassifyStep
from visionflow.core.pipeline.steps.inference.detection_router import RouteDetectionStep
from visionflow.core.pipeline.steps.inference.ocr import OcrStep
from visionflow.core.pipeline.steps.inference.detect import DetectStep
from visionflow.core.pipeline.steps.inference.filter import FilterStep
from visionflow.core.pipeline.steps.inference.replace_detection_image import ReplaceDetectionImageStep
from visionflow.core.pipeline.steps.transforms.binarize import BinarizeStep
from visionflow.core.pipeline.steps.transforms.resize import ResizeStep


class PipelineBuilder:
    def __init__(
        self,
        services: Dict[str, InferenceServiceBase],
        *,
        parent: Optional["PipelineBuilder"] = None,
    ):
        self._name = None
        self._in_key = None
        self._out_key = None
        self._services = services
        self._parent = parent
        self._steps: List[StepBase] = []
        self._in_router = False
        self._router_in_key = None
        self._router_out_key = None
        self._routes: Dict[str, StepBase] = {}

    def pipeline(self, name: str, in_key: str="default", out_key: str="default") -> "PipelineBuilder":
        self._name = name
        self._in_key = in_key
        self._out_key = out_key
        return self
    
    def binarize(self, normalize: bool=False, in_key: str="default", out_key: str="default") -> "PipelineBuilder":
        self._steps.append(BinarizeStep(normalize, in_key, out_key))
        return self
    
    def resize(self, reshape: Tuple[int, int], interpolation: int=cv2.INTER_CUBIC, in_key: str="default", out_key: str="default") -> "PipelineBuilder":
        self._steps.append(ResizeStep(reshape, interpolation, in_key, out_key))
        return self

    def detect(self, model_ref: str, in_key: str="default", out_key: str="default") -> "PipelineBuilder":
        svc = self._services[model_ref]
        self._steps.append(DetectStep(svc, in_key, out_key))
        return self
    
    def ocr(self, model_ref: str, in_key: str="default", out_key: str="default") -> "PipelineBuilder":
        svc = self._services[model_ref]
        self._steps.append(OcrStep(svc, in_key, out_key))
        return self

    def classify(self, model_ref: str, in_key: str="default", out_key: str="default") -> "PipelineBuilder":
        svc = self._services[model_ref]
        self._steps.append(ClassifyStep(svc, in_key, out_key))
        return self
    
    def filter(self, min_conf: float, in_key: str="default", out_key: str="default") -> "PipelineBuilder":
        self._steps.append(FilterStep(min_conf, in_key, out_key))
        return self
    
    def swap_image(self, exclude_detection: bool=False, in_key: str= "default", out_key: str="default") -> "PipelineBuilder":
        self._steps.append(ReplaceDetectionImageStep(exclude_detection=exclude_detection, in_key=in_key, out_key=out_key))
        return self

    def route_detection(self, in_key: str="default", out_key: str="default") -> "PipelineBuilder":
        self._in_router = True
        self._router_in_key = in_key
        self._router_out_key = out_key
        return self

    def route(self, name: str) -> "PipelineBuilder":
        if not self._in_router:
            raise ValueError("")
        
        route_builder = PipelineBuilder(
            services=self._services,
            parent=self
        ).pipeline(name=f"{self._name}::{name}")

        self._current_route = name
        return route_builder
    
    def end_route(self) -> "PipelineBuilder":
        if self._parent is None:
            raise ValueError("end_route() can only be called on an child builder")
        self._parent._routes[self._parent._current_route] = self.build()
        return self._parent
    
    def end_detection_router(self) -> "PipelineBuilder":
        self._steps.append(RouteDetectionStep(self._routes, self._router_in_key, self._router_out_key))
        self._reset_detection_router()
        return self
    
    def _reset_detection_router(self) -> None:
        self._in_router = False
        self._router_in_key = None
        self._router_out_key = None
        self._routes = {}
        self._current_route = None

    def _reset(self) -> None:
        self._steps
        self._name = None
        self._in_key = None
        self._out_key = None
        self._reset_detection_router()

    def build(self) -> Pipeline:
        pipeline = Pipeline(self._name, self._steps, self._in_key, self._out_key)
        validation = pipeline.validate()
        if not validation.ok:
            raise ValueError("")
        self._reset()
        return pipeline
