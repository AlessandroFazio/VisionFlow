import dataclasses
from typing import Dict, Callable

from prefect import task
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase
from visionflow.core.pipeline.utils.matchers import BranchMatcherBase
from visionflow.core.pipeline.utils.splitters import ExchangeSplitterBase


class SplitByStep(StepBase):
    def __init__(
        self,
        branches: Dict[str, StepBase],
        exchange_splitter: ExchangeSplitterBase,
        matcher: BranchMatcherBase
    ) -> None:
        self.branches = branches
        self.exchange_splitter = exchange_splitter
        self.matcher = matcher
        self.branch_keys = list(self.branches.keys())
        super().__init__(name="split_by")

    def _dispatch(
        self,
        context: PipelineContext,
        exchange: Exchange,
        executors: Dict[str, Callable[[PipelineContext, Exchange], Exchange]]
    ) -> Exchange:
        for ex in self.exchange_splitter.split(exchange):
            key, branch = self.matcher.match(ex, self.branch_keys)
            if not key:
                continue
            ex = dataclasses.replace(ex, execution_id=branch._execution_id())
            executor = executors[key]
            exchange = executor(context, ex)
        
        return exchange

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        python_routes = {n: route.process for n, route in self.branches.items()}
        return self._dispatch(context, exchange, python_routes)

    def to_prefect(self) -> task:
        prefect_routes = {n: step.to_prefect() for n, step in self.branches.items()}

        @task(name=self.name)
        def step_task(context: PipelineContext, exchange: Exchange) -> Exchange:
            return self._dispatch(context, exchange, prefect_routes)

        return step_task