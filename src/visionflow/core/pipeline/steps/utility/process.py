from typing import Callable

from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class ProcessStep(StepBase):
    def __init__(self, fn: Callable[[PipelineContext, Exchange], Exchange]) -> None:
        super().__init__(name=f"{self.__class__.__name__}[{fn.__name__}]")
        self.fn = fn

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        return self.fn(context, exchange)