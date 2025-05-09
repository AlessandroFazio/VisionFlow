from typing import Any, Dict, Optional
from visionflow.core.entity.factory import EntityFactory
from visionflow.core.pipeline.base import RuntimeOptions, StepBase, StepRunContext


class BuildEntityStep(StepBase):
    def __init__(
        self, 
        factory: EntityFactory, 
        runtime_options: Optional[RuntimeOptions]=None, 
        tags: Optional[Dict[str, Any]]=None
    ) -> None:
        super().__init__(runtime_options=runtime_options, tags=tags)
        self.factory = factory

    def process(self, context: StepRunContext) -> StepRunContext:
        entity = self.factory.from_exchange(context.exchange)
        context.entity_registry.register_entity(entity)
        return context