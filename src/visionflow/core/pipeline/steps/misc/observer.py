from typing import Callable

from visionflow.core.pipeline.base import Exchange, StepBase


class ObserverStep(StepBase):
    def __init__(self, callback: Callable[[Exchange], None]) -> None:
        self.callback = callback
        super().__init__(name=f"observer[{callback.__name__}]")

    def process(self, exchange: Exchange) -> Exchange:
        self.callback(exchange)
        return exchange