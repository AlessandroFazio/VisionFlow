from visionflow.core.entity.registry.base import EntityRegistryBase, EntityRegistryVisitorBase
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class ResolveEntityStep(StepBase):
    def __init__(self, resolver: EntityRegistryVisitorBase) -> None:
        super().__init__()
        self.resolver = resolver
    
    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        registry = context.get(EntityRegistryBase.pipeline_ctx_key(), EntityRegistryBase)
        registry.accept(self.resolver)
        return exchange