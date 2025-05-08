from visionflow.core.pipeline.builder.pipeline import PipelineBuilder


class VisionFlow:
    @staticmethod
    def pipeline(pipeline_name: str) -> PipelineBuilder:
        builder = PipelineBuilder(name=pipeline_name)
        return builder