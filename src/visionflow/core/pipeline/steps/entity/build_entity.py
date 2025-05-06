from visionflow.core.entity.factory import EntityFactory
from visionflow.core.entity.registry.base import EntityRegistryBase
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class BuildEntityStep(StepBase):
    def __init__(self, factory: EntityFactory) -> None:
        super().__init__()
        self.factory = factory

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        registry = context.get(EntityRegistryBase.pipeline_ctx_key(), EntityRegistryBase)
        entity = self.factory.from_exchange(exchange)
        registry.register_entity(entity)
        return exchange