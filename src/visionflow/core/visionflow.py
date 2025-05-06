from dataclasses import Field
from typing import List

from pydantic import BaseModel

from visionflow.core.inference.config import ModelConfig
from visionflow.core.inference.factory import InferenceServiceFactory
from visionflow.core.pipeline.builder.pipeline import PipelineBuilder


class VisionFlowConfig(BaseModel):
    models: List[ModelConfig] = Field(default_factory=list)


class VisionFlow:
    def __init__(self, config: VisionFlowConfig) -> None:
        self.config = config

    def builder(self, pipeline_name: str) -> PipelineBuilder:
        builder = PipelineBuilder(
            name=pipeline_name,
            services={m.name: InferenceServiceFactory.create(m) for m in self.config.models}
        )
        return builder