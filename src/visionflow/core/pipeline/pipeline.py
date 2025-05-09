from typing import Any, Dict, List, Optional

from visionflow.core.pipeline.base import CompositeStep, Exchange, RuntimeOptions, StepBase, StepRunContext


class Pipeline(CompositeStep):
    def __init__(
        self, 
        name: str, 
        steps: List[StepBase], 
        runtime_options: Optional[RuntimeOptions]=None,
        tags: Optional[Dict[str, Any]]=None
    ) -> None:
        super().__init__(name=name, runtime_options=runtime_options, tags=tags)
        self._steps = steps
    
    def steps(self) -> List[StepBase]:
        return self._steps
    
    def rebuild_with_steps(self, steps: List[StepBase]) -> "CompositeStep":
        return Pipeline(self.name, steps, self.runtime_options, self.tags)
    
    def run(self, context: StepRunContext) -> Exchange:
        return self.process(context)
    
    def process(self, context: StepRunContext) -> StepRunContext:
        for step in self._steps:
            result = context.dispatcher.submit(step, context)
            exchange = result.get()
        return exchange
    
    def explain(self, depth: int = 0) -> str:
        lines = [f"Pipeline: {self.name}"]
        for step in self._steps:
            lines.append(step.explain(depth + 1))
        return "\n".join(lines)

