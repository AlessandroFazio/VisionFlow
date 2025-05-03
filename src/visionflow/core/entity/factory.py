from typing import Dict, Type

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.processors import ClassificationLabelProcessor, EntityRefProcessor, FieldProcessorBase, OcrRegexProcessor
from visionflow.core.entity.reflection.meta import FieldType
from visionflow.core.pipeline.base import Exchange


class EntityFactory:
    def __init__(self) -> None:
        self._processors: Dict[FieldType, FieldProcessorBase] = {
            FieldType.OCR_REGEX: OcrRegexProcessor(),
            FieldType.CLASSIFICATION_LABEL: ClassificationLabelProcessor(),
            FieldType.ENTITY_REF: EntityRefProcessor()
        }

    def from_exchange(self, cls: Type[EntityBase], exchange: Exchange) -> EntityBase:
        fields = cls.__vf_meta__.fields
        kwargs = {}
        for fname, meta in fields.items():
            processor = self._processors[meta.field_type]
            kwargs[fname] = processor.process(meta, exchange)
        return cls(**kwargs)
    