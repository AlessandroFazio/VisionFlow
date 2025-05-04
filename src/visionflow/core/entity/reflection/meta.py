from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Type, Union

if TYPE_CHECKING:
    from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.reflection.types import ClassificationLabelConverter, OcrRegexConverter


@dataclass
class OcrRegexConfig:
    pattern: str
    line_match: bool
    stop_on_first_match: bool

@dataclass
class ClassificationLabelConfig:
    allowed_labels: Optional[List[str]] = None

class FieldType(Enum):
    OCR_REGEX = 0
    CLASSIFICATION_LABEL = 1
    ENTITY_REF = 2

@dataclass
class FieldMeta:
    field_type: FieldType
    type_hint: Any
    priority: int
    converter: Union[OcrRegexConverter | ClassificationLabelConverter]
    config: Union[ClassificationLabelConfig | OcrRegexConfig | None] = None

@dataclass
class EntityMeta:
    branch_binding: Optional[str] = None
    fields: Dict[str, FieldMeta] = field(default_factory=dict)