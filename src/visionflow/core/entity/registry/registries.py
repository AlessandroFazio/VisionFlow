from collections import defaultdict
from threading import RLock
from typing import Dict, Iterable, List, Type

from visionflow.collections.hierarchical_map import HierarchicalMap
from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.registry.base import EntityRegistryBase, EntityRegistryVisitorBase
from visionflow.core.entity.utils import Entity


class HierarchicalEntityRegistry(EntityRegistryBase):
    def __init__(
        self,
        class_hierarchy: HierarchicalMap[str, Type[EntityBase]]
    ) -> None:
        self._class_hierarchy = class_hierarchy
        self._entities: Dict[str, List[EntityBase]] = defaultdict(list)
        self._lock = RLock()

    def register(self, entity: EntityBase) -> None:
        with self._lock:
            name = Entity.name(entity.__class__)
            if not name in self._class_hierarchy:
                raise ValueError("")
            self._entities[name].append(entity)

    def entities(self) -> Iterable[EntityBase]:
        with self._lock:
            return list(self._entities.values())
        
    def entities_by_class(self, entity_cls: Type[EntityBase]) -> Iterable[EntityBase]:
        with self._lock:
            return self._entities.get(Entity.name(entity_cls), [])
        
    def parent_candidates(self, entity: EntityBase) -> Iterable[EntityBase]:
        entity_name = Entity.name(entity.__class__)
        return [
            self.entities_by_class(Entity.name(c)) 
            for c in self._class_hierarchy.parents_of(entity_name)
        ]

    def accept(self, visitor: "EntityRegistryVisitorBase") -> None:
        with self._lock:
            visitor.visit(self)

    @classmethod
    def from_root_class(cls, entity_cls: Type[EntityBase]) -> "HierarchicalEntityRegistry":
        return cls(
            HierarchicalMap(
                hierarchy=entity_cls,
                key_fn=Entity.name, 
                children_fn=lambda e: (m.type_info.base_type for (_,m) in Entity.iter_children(e))
            )
        )