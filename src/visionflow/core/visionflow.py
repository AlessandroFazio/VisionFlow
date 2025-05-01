from dataclasses import Field
from typing import List

from pydantic import BaseModel

from visionflow.core.inference.config import ModelConfig
from visionflow.core.inference.factory import InferenceServiceFactory
from visionflow.core.pipeline.builder import PipelineBuilder


class VisionFlowConfig(BaseModel):
    models: List[ModelConfig] = Field(default_factory=list)


class VisionFlow:
    @classmethod
    def pipeline_builder(cls, config: VisionFlowConfig) -> PipelineBuilder:
        builder = PipelineBuilder(
            services={InferenceServiceFactory.create(m) for m in config.models}
        )
        return builder
