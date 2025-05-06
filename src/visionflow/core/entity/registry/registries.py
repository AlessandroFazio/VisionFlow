from collections import defaultdict
from threading import RLock
from typing import Dict, Iterable, List, Type

import networkx as nx 

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.iterator.hierarchy import EntityHierarchyIterator
from visionflow.core.entity.registry.base import EntityRegistryBase, EntityRegistryVisitorBase
from visionflow.core.entity.utils import Entity


class GraphEntityRegistry(EntityRegistryBase):
    def __init__(
        self
    ) -> None:
        self._class_graph = nx.DiGraph()
        self._entities: Dict[str, List[EntityBase]] = defaultdict(list)
        self._lock = RLock()
            
    def register_class(self, entity_cls: Type[EntityBase]) -> None:

        def visit(entity_cls: Type[EntityBase]) -> bool:
            changed = False
            name = Entity.name(entity_cls)
            if name in self._class_graph.nodes:
                return changed
            self._class_graph.add_node(name, cls=entity_cls)
            for _, child_m in Entity.iter_children(entity_cls):
                child_cls = child_m.type_info.base_type
                self._class_graph.add_edge(name, Entity.name(child_cls))
                changed |= visit(child_cls)
            return changed
        
        with self._lock:
            if visit(entity_cls) and not nx.is_directed_acyclic_graph(self._class_graph):
                raise ValueError("")
            
    def parents_classes(self, entity_cls: Type[EntityBase]) -> List[Type[EntityBase]]:
        with self._lock:
            return [e[0] for e in self._class_graph.in_edges(Entity.name(entity_cls))]

    def register_entity(self, entity: EntityBase) -> None:
        with self._lock:
            name = Entity.name(entity.__class__)
            if not name in self._class_graph.nodes:
                raise ValueError("")
            self._entities[name].append(entity)

    def entities(self) -> Iterable[EntityBase]:
        with self._lock:
            return list(self._entities.values())
        
    def entities_for_class(self, entity_cls: Type[EntityBase]) -> Iterable[EntityBase]:
        with self._lock:
            return self._entities.get(Entity.name(entity_cls), [])
        
    def parent_candidates(self, entity: EntityBase) -> Iterable[EntityBase]:
        with self._lock:
            return [
                self.entities_for_class(c)
                for c in self.parents_classes(entity.__class__)
            ]

    def accept(self, visitor: "EntityRegistryVisitorBase") -> None:
        with self._lock:
            visitor.visit(self)