from typing import Dict, Callable

from prefect import task
from visionflow.core.pipeline.base import Exchange, StepBase
from visionflow.core.pipeline.utils.matchers import BranchMatcherBase
from visionflow.core.pipeline.utils.multiplexers import ExchangeMultiplexerBase


class SplitByStep(StepBase):
    def __init__(
        self,
        branches: Dict[str, StepBase],
        multiplexer: ExchangeMultiplexerBase,
        matcher: BranchMatcherBase
    ) -> None:
        self.branches = branches
        self.multiplexer = multiplexer
        self.matcher = matcher
        self.branch_keys = list(self.branches.keys())
        super().__init__(name="split_by")

    def _dispatch(
        self,
        exchange: Exchange,
        executors: Dict[str, Callable[[Exchange], Exchange]]
    ) -> Exchange:
        for ex in self.multiplexer.multiplex(exchange):
            key, branch = self.matcher.match(ex, self.branch_keys)
            if not key:
                continue

            exec_id = branch._execution_id()
            exchange.execution_id = exec_id
            executor = executors[key]
            exchange.children[exec_id] = executor(ex)
        
        return exchange

    def process(self, exchange: Exchange) -> Exchange:
        python_routes = {n: route.process for n, route in self.branches.items()}
        return self._dispatch(exchange, python_routes)

    def to_prefect(self) -> task:
        prefect_routes = {n: step.to_prefect() for n, step in self.branches.items()}

        @task(name=self.name)
        def step_task(exchange: Exchange) -> Exchange:
            return self._dispatch(exchange, prefect_routes)

        return step_task