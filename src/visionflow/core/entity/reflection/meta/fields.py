from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Type, Union, get_args, get_origin

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.reflection.types import ClassificationLabelConverter, EntityRefConverter, OcrRegexConverter
from visionflow.core.entity.sort.base import EntitySortStrategy


@dataclass
class FieldTypeInfo:
    raw_type: Any
    base_type: Type
    is_list: bool
    is_optional: bool
    elements_optional: bool

    @classmethod
    def from_type_hint(cls, hint: Any) -> "FieldTypeInfo":
        original_hint = hint
        is_optional = False
        is_list = False
        elements_optional = False

        if get_origin(hint) is Union:
            args = get_args(hint)
            if type(None) in args:
                non_none = [arg for arg in args if arg is not type(None)]
                if len(non_none) == 1:
                    hint = non_none[0]
                    is_optional = True

        if get_origin(hint) in (list, List):
            is_list = True
            inner_hint = get_args(hint)[0]

            if get_origin(inner_hint) is Union and type(None) in get_args(inner_hint):
                elements_optional = True
                base_type = [arg for arg in get_args(inner_hint) if arg is not type(None)][0]
            else:
                base_type = inner_hint

        else:
            base_type = hint

        return cls(
            raw_type=original_hint,
            base_type=base_type,
            is_list=is_list,
            is_optional=is_optional,
            elements_optional=elements_optional
        )
        

class FieldType(Enum):
    OCR_REGEX = 0
    CLASSIFICATION_LABEL = 1
    ENTITY_REF = 2
    

@dataclass
class OcrRegexConfig:
    rule_selector: Optional[List[str]] = None
    match_key: Optional[str] = None


@dataclass
class ClassificationLabelConfig:
    allowed_labels: Optional[List[str]] = None


@dataclass
class EntityRefConfig:
    entity_cls: Type[EntityBase]
    sort_strategy: Optional[EntitySortStrategy] = None


@dataclass
class FieldMeta:
    field_type: FieldType
    type_info: FieldTypeInfo
    priority: int
    converter: Union[OcrRegexConverter | ClassificationLabelConverter | EntityRefConverter]
    config: Union[ClassificationLabelConfig | OcrRegexConfig | EntityRefConfig | None] = None