from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Type, get_type_hints

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.reflection.meta import ClassificationLabelConfig, FieldMeta, FieldType, OcrRegexConfig
from visionflow.core.entity.reflection.types import ClassificationLabelConverter, OcrRegexConverter, default_single_label_converter


class FieldBase(ABC):
    def __init__(
        self,
        priority: int=0
    ) -> None:
        self.priority = priority
        self.name = None

    @abstractmethod
    def __to_field_meta(self, type_hint: Any) -> FieldMeta:
        pass
    
    def __field_type_hint(self, owner: Type[EntityBase], name: str) -> Any:
        raw = owner.__annotations__.get(name)
        hints = get_type_hints(owner)
        return hints.get(name, raw)

    def __register_field_meta(self, owner: Type[EntityBase], name: str) -> None:
        type_hint = self.__field_type_hint(owner, name)
        owner.__vf_meta__.fields[name] = self.__to_field_meta(type_hint)

    def __set_name__(self, owner: Type[EntityBase], name: str) -> None:
        self.name = name
        self.__register_field_meta(owner, name)

    def __set__(self, instance: EntityBase, value: Any) -> None:
        if instance is None:
            return        
        instance.__dict__[self.name] = value

    def __get__(self, instance: EntityBase, owner: Type[EntityBase]) -> Any:
        return instance.__dict__.get(self.name) if instance else self

class ClassificationLabelField(FieldBase):
    def __init__(
        self, 
        allowed_labels: List[str]=None, 
        converter: Optional[ClassificationLabelConverter]=None, 
        priority: int=0
    ) -> None:
        self.allowed_labels = allowed_labels
        self.converter = converter or default_single_label_converter
        super().__init__(priority=priority)
    
    def __to_field_meta(self, type_hint: Any) -> FieldMeta:
        return FieldMeta(
            field_type=FieldType.CLASSIFICATION_LABEL,
            config=ClassificationLabelConfig(
                allowed_labels=self.allowed_labels
            ),
            type_hint=type_hint,
            converter=self.converter,
            priority=self.priority
        )

class OcrRegexField(FieldBase):
    def __init__(
        self, 
        pattern: str,
        converter: OcrRegexConverter,
        line_match: bool=False,
        first_match_only: bool=False,
        priority: int=0
    ) -> None:
        self.pattern = pattern
        self.converter = converter
        self.line_match = line_match
        self.first_match_only = first_match_only
        super().__init__(priority=priority)

    def __to_field_meta(self, type_hint: Any) -> FieldMeta:
        return FieldMeta(
            field_type=FieldType.OCR_REGEX,
            config=OcrRegexConfig(
                pattern=self.pattern
            ),
            type_hint=type_hint,
            converter=self.converter,
            priority=self.priority
        )

class EntityRefField(FieldBase):
    def __init__(
        self, 
        converter: Optional[Callable[[Any], Any]]=None, 
        priority: int=0
    ) -> None:
        self.converter = converter
        super().__init__(priority=priority)

    def __to_field_meta(self, type_hint: Any) -> FieldMeta:
        return FieldMeta(
            field_type=FieldType.ENTITY_REF,
            type_hint=type_hint,
            converter=self.converter,
            priority=self.priority
        )
