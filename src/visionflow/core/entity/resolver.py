from visionflow.core.entity.base import Entity
from visionflow.core.entity.registry import EntityRegistry


class EntityRefResolverVisitor:
    def visit(self, registry: EntityRegistry) -> None:
        for name, entity_cls in registry.class_hierarchy:
            meta = Entity.meta(entity_cls)
            if meta.
            instances = registry.instances_by_entity_name(name)
