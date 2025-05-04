from typing import Dict, Type

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.processors import ClassificationLabelProcessor, EntityRefProcessor, FieldProcessorBase, OcrRegexProcessor
from visionflow.core.entity.reflection.meta import FieldType
from visionflow.core.pipeline.base import Exchange


class EntityFactory:
    _processors: Dict[FieldType, FieldProcessorBase] = {
            FieldType.OCR_REGEX: OcrRegexProcessor(),
            FieldType.CLASSIFICATION_LABEL: ClassificationLabelProcessor(),
            FieldType.ENTITY_REF: EntityRefProcessor()
        }
    
    def __init__(self, entity_cls: Type[EntityBase]) -> None:
        self._entity_cls = entity_cls

    def from_exchange(self, exchange: Exchange) -> EntityBase:
        fields = self._entity_cls.__vf_meta__.fields
        kwargs = {}
        for fname, meta in fields.items():
            processor = self._processors[meta.field_type]
            kwargs[fname] = processor.process(meta, exchange)
        return self._entity_cls(**kwargs)
    