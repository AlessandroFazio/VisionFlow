from abc import ABC, abstractmethod
from typing import Any, List, Optional, Type, Union

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.reflection.meta import ClassificationLabelConfig, EntityRefConfig, FieldMeta, FieldType, FieldTypeInfo, OcrRegexConfig
from visionflow.core.entity.reflection.types import ClassificationLabelConverter, EntityRefConverter, OcrRegexConverter, default_single_label_converter
from visionflow.core.entity.sort.config import SortConfig, SortType
from visionflow.core.entity.sort.factory import EntitySortFactory


class FieldBase(ABC):
    def __init__(
        self,
        priority: int=0
    ) -> None:
        self.priority = priority
        self.name = None

    @abstractmethod
    def _to_field_meta(self, type_info: FieldTypeInfo) -> FieldMeta:
        pass

    def __set_name__(self, owner: Type[EntityBase], name: str) -> None:
        self.name = name

    def __set__(self, instance: EntityBase, value: Any) -> None:
        if instance is None:
            return        
        instance.__dict__[self.name] = value

    def __get__(self, instance: EntityBase, owner: Type[EntityBase]) -> Any:
        return instance.__dict__.get(self.name) if instance else self

class ClassificationLabel(FieldBase):
    def __init__(
        self, 
        allowed_labels: List[str]=None, 
        converter: Optional[ClassificationLabelConverter]=None, 
        priority: int=0
    ) -> None:
        self.allowed_labels = allowed_labels
        self.converter = converter or default_single_label_converter
        super().__init__(priority=priority)
    
    def _to_field_meta(self, type_info: FieldTypeInfo) -> FieldMeta:
        return FieldMeta(
            field_type=FieldType.CLASSIFICATION_LABEL,
            config=ClassificationLabelConfig(
                allowed_labels=self.allowed_labels
            ),
            type_info=type_info,
            converter=self.converter,
            priority=self.priority
        )

class OcrRegex(FieldBase):
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

    def _to_field_meta(self, type_info: FieldTypeInfo) -> FieldMeta:
        return FieldMeta(
            field_type=FieldType.OCR_REGEX,
            config=OcrRegexConfig(
                pattern=self.pattern
            ),
            type_info=type_info,
            converter=self.converter,
            priority=self.priority
        )

class EntityRef(FieldBase):
    def __init__(
        self,
        entity_cls: Optional[Type[EntityBase]]=None,
        sort_type: Union[str | SortType | None]=None,
        sort_keys: Optional[List[str]]=None,
        converter: Optional[EntityRefConverter]=None, 
        priority: int=0
    ) -> None:
        self.entity_cls = entity_cls
        self.sort_type = sort_type
        self.sort_keys = sort_keys
        self.converter = converter or (lambda x: x)
        super().__init__(priority=priority)

    def _to_field_meta(self, type_info: FieldTypeInfo) -> FieldMeta:
        sort_type = self.sort_type
        if isinstance(sort_type, str):
            sort_type = SortType.from_value(sort_type)
        
        sort_strategy = None
        if sort_type:
            sort_config = SortConfig(sort_type, self.sort_keys)
            sort_strategy = EntitySortFactory.create(sort_config)

        return FieldMeta(
            field_type=FieldType.ENTITY_REF,
            type_info=type_info,
            converter=self.converter,
            priority=self.priority,
            config=EntityRefConfig(
                entity_cls=self.entity_cls,
                sort_strategy=sort_strategy
            )
        )
