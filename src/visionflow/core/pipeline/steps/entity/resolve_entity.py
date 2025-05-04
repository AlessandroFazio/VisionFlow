from visionflow.core.entity.registry.base import EntityRegistryBase, EntityRegistryVisitorBase
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class ResolveEntityStep(StepBase):
    def __init__(self, resolver: EntityRegistryVisitorBase) -> None:
        self.resolver = resolver
        super().__init__(name="resolve_entity")
    
    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        registry = context.get(EntityRegistryBase.pipeline_ctx_key(), EntityRegistryBase)
        registry.accept(self.resolver)
        return exchange