from typing import Callable

from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class ProcessStep(StepBase):
    def __init__(self, fn: Callable[[PipelineContext, Exchange], Exchange]) -> None:
        self.fn = fn
        super().__init__(name=f"observer[{fn.__name__}]")

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        return self.fn(context, exchange)