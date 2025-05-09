from visionflow.core.pipeline.backend.base import BackendBase
from visionflow.core.pipeline.backend.prefect import PrefectBackend
from visionflow.core.pipeline.backend.python import PythonBackend
from visionflow.core.pipeline.builder.context import RunContextBuilder
from visionflow.core.pipeline.builder.pipeline import PipelineBuilder


class VisionFlow:
    @staticmethod
    def pipeline(pipeline_name: str) -> PipelineBuilder:
        builder = PipelineBuilder(name=pipeline_name)
        return builder
    
    @staticmethod
    def run_context() -> RunContextBuilder:
        return RunContextBuilder()
    
    @staticmethod
    def python_backend() -> BackendBase:
        return PythonBackend()
    
    @staticmethod
    def prefect_backend() -> BackendBase:
        return PrefectBackend()