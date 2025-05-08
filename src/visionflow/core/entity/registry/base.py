from abc import ABC, abstractmethod
from typing import Iterable, List, Type

from visionflow.core.entity.base import EntityBase


class EntityRegistryVisitorBase(ABC):
    @abstractmethod
    def visit(self, registry: "EntityRegistryBase") -> None:
        pass


class EntityRegistryBase(ABC):
    @abstractmethod
    def register_class(self, entity_cls: Type[EntityBase]) -> None:
        pass

    @abstractmethod
    def register_entity(self, entity: EntityBase) -> None:
        pass

    @abstractmethod
    def entities(self) -> Iterable[EntityBase]:
        pass
        
    @abstractmethod
    def entities_for_class(self, entity_cls: Type[EntityBase]) -> Iterable[EntityBase]:
        pass

    @abstractmethod
    def parent_candidates(self, entity: EntityBase) -> List[EntityBase]:
        pass

    @abstractmethod
    def accept(self, visitor: "EntityRegistryVisitorBase") -> None:
        pass
    
    @abstractmethod
    def parents_classes(self, entity_cls: Type[EntityBase]) -> List[Type[EntityBase]]:
        pass

    @classmethod
    def pipeline_ctx_key(cls) -> str:
        return '__vf_entity_registry__'