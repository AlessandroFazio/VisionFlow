from typing import Any, Callable, Dict, Optional

from visionflow.core.pipeline.base import RuntimeOptions, StepBase, StepRunContext


class ProcessStep(StepBase):
    def __init__(
        self, 
        fn: Callable[[StepRunContext], StepRunContext],
        runtime_options: Optional[RuntimeOptions]=None, 
        tags: Optional[Dict[str, Any]]=None
    ) -> None:
        super().__init__(
            name=f"{self.__class__.__name__}[{fn.__name__}]", 
            runtime_options=runtime_options, 
            tags=tags
        )
        self.fn = fn

    def process(self, context: StepRunContext) -> StepRunContext:
        return self.fn(context)