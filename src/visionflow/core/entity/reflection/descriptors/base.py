from abc import ABC, abstractmethod
from typing import Any, Type

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.reflection.meta.field import FieldMeta
from visionflow.core.pipeline.base import Exchange


class FieldBase(ABC):
    def __init__(self) -> None:
        self.name = None

    def __set_name__(self, owner: Type[EntityBase], name: str) -> None:
        self.name = name

    def __set__(self, instance: EntityBase, value: Any) -> None:
        if instance is None:
            return        
        instance.__dict__[self.name] = value

    def __get__(self, instance: EntityBase, owner: Type[EntityBase]) -> Any:
        return instance.__dict__.get(self.name) if instance else self


class FieldProcessorBase(ABC):
    @abstractmethod
    def process(self, field: FieldMeta, exchange: Exchange) -> Any:
        pass