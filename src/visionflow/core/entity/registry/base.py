from abc import ABC, abstractmethod
from typing import Iterable, Type

from visionflow.core.entity.base import EntityBase


class EntityRegistryVisitorBase(ABC):
    @abstractmethod
    def visit(self, registry: "EntityRegistryBase") -> None:
        pass


class EntityRegistryBase(ABC):
    @abstractmethod
    def register(self, entity: EntityBase) -> None:
        pass

    @abstractmethod
    def entities(self) -> Iterable[EntityBase]:
        pass
        
    @abstractmethod
    def entities_by_class(self, entity_cls: Type[EntityBase]) -> Iterable[EntityBase]:
        pass

    @abstractmethod
    def parent_candidates(self, entity: EntityBase) -> Iterable[EntityBase]:
        pass

    @abstractmethod
    def accept(self, visitor: "EntityRegistryVisitorBase") -> None:
        pass

    @classmethod
    def pipeline_ctx_key(cls) -> str:
        return '__vf_entity_registry__'