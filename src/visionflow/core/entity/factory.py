from typing import Dict, Type

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.reflection.descriptors.processors import ClassificationLabelProcessor, EntityRefProcessor, FieldProcessorBase, OcrRegexProcessor
from visionflow.core.entity.reflection.meta.entity import EntityType
from visionflow.core.entity.reflection.meta.field import FieldType
from visionflow.core.entity.utils import Entity
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
        entity_meta = Entity.meta(self._entity_cls)
        coordinates, kwargs = None, {}
        
        for fname, meta in entity_meta.fields.items():
            processor = self._processors[meta.field_type]
            kwargs[fname] = processor.process(meta, exchange)
        
        if entity_meta.source.type == EntityType.PHYSICAL:
            coordinates = entity_meta.source.provider(exchange)
            img_shape = exchange.original_image_shape

        return self._entity_cls(
            __vf_xy__=coordinates, 
            __vf_img_shape__=img_shape, 
            **kwargs
        )
    