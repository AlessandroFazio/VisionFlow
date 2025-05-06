from typing import List
from collections.abc import Iterable
from visionflow.core.entity.reflection.meta import FieldType
from visionflow.core.entity.registry.base import EntityRegistryBase, EntityRegistryVisitorBase
from visionflow.core.entity.utils import Entity


class ParentEntityResolver(EntityRegistryVisitorBase):
    def visit(self, registry: EntityRegistryBase) -> None:
        for entity in registry.entities():
            meta = Entity.meta(entity.__class__)
            if not meta.parent_selector:
                continue
            
            candidates = registry.parent_candidates(entity)
            selected_parent = meta.parent_selector.select(entity, candidates)
            if not selected_parent:
                continue

            for fname, child_meta in Entity.iter_children(selected_parent.__class__):
                if not isinstance(entity, child_meta.type_info.base_type):
                    continue

                value = getattr(selected_parent, fname, None)
                entity_val = child_meta.converter(entity)
                
                if child_meta.type_info.is_list:
                    if isinstance(value, list):
                        value.append(entity_val)
                    else:
                        value = [entity_val]
                else:
                    value = entity_val
                setattr(selected_parent, fname, value)
                break


class SortedEntityResolver(EntityRegistryVisitorBase):
    def visit(self, registry: EntityRegistryBase) -> None:
        for entity in registry.entities():          
            for fname, meta in Entity.iter_children(entity.__class__):
                if meta.config.sort_strategy is None:
                    continue
                
                child_entities = getattr(entity, fname, None)
                if not isinstance(child_entities, Iterable) or len(child_entities) < 2:
                    continue

                sorted = meta.config.sort_strategy.argsort(child_entities)
                if sorted:
                    setattr(entity, fname, [child_entities[i] for i in sorted])


class EntityRegistryResolver(EntityRegistryVisitorBase):
    def __init__(self, visitors: List[EntityRegistryVisitorBase]) -> None:
        self._visitors = visitors

    def visit(self, registry: EntityRegistryBase) -> None:
        for visitor in self._visitors:
            visitor.visit(registry)

    @classmethod
    def default(cls) -> EntityRegistryVisitorBase:
        return cls(
            visitors=[
                ParentEntityResolver(),
                SortedEntityResolver()
            ]
        )