from collections import defaultdict
from threading import RLock
from typing import Dict, List, Type

from visionflow.collections.hierarchical_map import HierarchicalMap
from visionflow.core.entity.base import Entity, EntityBase
from visionflow.core.entity.resolver import EntityRefResolverVisitor


class EntityRegistry:
    def __init__(
        self,
        class_hierarchy: HierarchicalMap[str, Type[EntityBase]]
    ) -> None:
        self.class_hierarchy = class_hierarchy
        self._instances: Dict[str, List[EntityBase]] = defaultdict(list)
        self._lock = RLock()

    def register(self, entity: EntityBase) -> None:
        with self._lock:
            name = Entity.name(entity.__class__)
            if not name in self.class_hierarchy:
                raise ValueError("")
            self._instances[name].append(entity)

    def instances(self) -> List[EntityBase]:
        with self._lock:
            return list(self._instances.values())
        
    def instances_by_entity_name(self, name: str) -> List[EntityBase]:
        with self._lock:
            return self._instances.get(name, [])

    def accept(self, visitor: EntityRefResolverVisitor) -> None:
        with self._lock:
            visitor.visit(self)

    @classmethod
    def from_root_cls(cls, root: Type[EntityBase]) -> "EntityRegistry":
        return cls(HierarchicalMap(
            hierarchy=root, 
            key_fn=Entity.name, 
            children_fn=Entity.iter_children
        ))
    
    @classmethod
    def pipeline_ctx_key(cls) -> str:
        return '__vf_entity_registry__'