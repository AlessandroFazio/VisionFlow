from typing import Callable

from visionflow.core.pipeline.base import Exchange, StepBase


class ApplyStep(StepBase):
    def __init__(self, fn: Callable[[Exchange], Exchange]) -> None:
        self.fn = fn
        super().__init__(name=f"observer[{fn.__name__}]")

    def process(self, exchange: Exchange) -> Exchange:
        return self.fn(exchange)