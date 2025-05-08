from visionflow.core.entity.factory import EntityFactory
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase


class BuildEntityStep(StepBase):
    def __init__(self, factory: EntityFactory) -> None:
        super().__init__()
        self.factory = factory

    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        entity = self.factory.from_exchange(exchange)
        exchange.entity_registry.register_entity(entity)
        return exchange