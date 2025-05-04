from visionflow.core.entity.factory import EntityFactory
from visionflow.core.entity.registry.base import EntityRegistryBase
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class BuildEntityStep(StepBase):
    def __init__(self, factory: EntityFactory) -> None:
        self.factory = factory
        super().__init__(name="build_entity")

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        registry = context.get(EntityRegistryBase.pipeline_ctx_key(), EntityRegistryBase)
        entity = self.factory.from_exchange(exchange)
        registry.register(entity)
        return exchange