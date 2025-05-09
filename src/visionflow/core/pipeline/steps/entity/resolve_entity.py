from typing import Any, Dict, Optional
from visionflow.core.entity.registry.base import EntityRegistryVisitorBase
from visionflow.core.pipeline.base import RuntimeOptions, StepBase, StepRunContext


class ResolveEntityStep(StepBase):
    def __init__(
        self, 
        resolver: EntityRegistryVisitorBase,
        runtime_options: Optional[RuntimeOptions]=None, 
        tags: Optional[Dict[str, Any]]=None
    ) -> None:
        super().__init__(runtime_options=runtime_options, tags=tags)
        self.resolver = resolver
    
    def process(self, context: StepRunContext) -> StepRunContext:
        context.entity_registry.accept(self.resolver)
        return context