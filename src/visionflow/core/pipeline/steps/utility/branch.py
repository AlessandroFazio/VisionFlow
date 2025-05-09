import dataclasses
from typing import Any, Dict, List, Optional

from visionflow.core.pipeline.backend.base import DispatcherResultBase
from visionflow.core.pipeline.base import CompositeStep, RuntimeOptions, StepBase, StepRunContext
from visionflow.core.pipeline.utils.selector import StepSelectorBase
from visionflow.core.pipeline.utils.splitters import ExchangeSplitterBase, ZeroSplitter


class BranchStep(CompositeStep):
    def __init__(
        self,
        steps: List[StepBase],
        selector: StepSelectorBase,
        splitter: Optional[ExchangeSplitterBase]=None,
        runtime_options: Optional[RuntimeOptions]=None, 
        tags: Optional[Dict[str, Any]]=None
    ) -> None:
        super().__init__(runtime_options=runtime_options, tags=tags)
        self._steps = steps
        self.selector = selector
        self.splitter = splitter or ZeroSplitter()
    
    def steps(self) -> List[StepBase]:
        return self._steps
    
    def rebuild_with_steps(self, steps: List[StepBase]) -> "CompositeStep":
        return BranchStep(
            steps=steps,
            splitter=self.splitter,
            selector=self.selector
        )

    def process(self, context: StepRunContext) -> StepRunContext:
        results: List[DispatcherResultBase] = []
        for ex in self.splitter.split(context.exchange):
            for step in self.selector.select(ex, self._steps):
                ctx_ = dataclasses.replace(context, exchange=ex)
                results.append(context.dispatcher.submit(step, ctx_))
        
        _ = [r.get() for r in results]
        return context
    
    def explain(self, depth: int = 0) -> str:
        lines = [f"{'  ' * depth}- {self.name} (SplitBy)"]
        for step in self._steps:
            lines.append(f"{'  ' * (depth + 1)}[branch: {step.name}]")
            lines.append(step.explain(depth + 2))
        return "\n".join(lines)