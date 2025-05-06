from typing import Dict, Iterator, Tuple, Type
from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.reflection.meta.entity import EntityMeta, FieldMeta, FieldType


class Entity:
    @staticmethod
    def name(entity_cls: Type["EntityBase"]) -> str:
        return entity_cls.__name__
    
    @staticmethod
    def meta(entity_cls: Type["EntityBase"]) -> EntityMeta:
        return entity_cls.__vf_meta__
    
    @staticmethod
    def iter_children(entity_cls: Type["EntityBase"]) -> Iterator[Tuple[str, FieldMeta]]:
        fields = entity_cls.__vf_meta__.fields
        for meta in fields.values():
            if meta.field_type == FieldType.ENTITY_REF:
                yield meta