import sys
from typing import Callable, Dict, Optional, Tuple, Type, Union, get_type_hints
from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.parent_selector.config import ParentSelectorType
from visionflow.core.entity.parent_selector.factory import ParentSelectorFactory
from visionflow.core.entity.reflection.descriptors.base import FieldBase
from visionflow.core.entity.reflection.descriptors.fields import ClassificationLabel, EntityRef, OcrRegex
from visionflow.core.entity.reflection.meta.entity import EntityMeta, EntitySource, EntityType
from visionflow.core.entity.reflection.meta.field import ClassificationLabelConfig, EntityRefConfig, FieldMeta, FieldType, FieldTypeInfo, OcrRegexConfig
from visionflow.core.entity.sort.config import SortType
from visionflow.core.entity.sort.factory import EntitySortFactory
from visionflow.core.pipeline.utils.providers import DetectionCoordinatesProvider, StaticCoordinatesProvider


class FieldMetaFactory:
    @classmethod
    def create(cls, desc: FieldBase, type_info: FieldTypeInfo) -> FieldMeta:
        handler = cls._handlers[desc.__class__]
        return handler(desc, type_info)
    
    @staticmethod 
    def _create_entity_ref_meta(desc: EntityRef, type_info: FieldTypeInfo) -> FieldMeta:
        sort_type = desc.sort_type
        if isinstance(sort_type, str):
            sort_type = SortType.from_value(sort_type)
        
        sort_strategy = None
        if sort_type:
            sort_strategy = EntitySortFactory.create(sort_type, desc.sort_keys)

        return FieldMeta(
            field_type=FieldType.ENTITY_REF,
            type_info=type_info,
            converter=desc.converter,
            config=EntityRefConfig(
                entity_cls=desc.entity_cls,
                sort_strategy=sort_strategy
            )
        )
    
    @staticmethod
    def _create_classification_label_meta(desc: ClassificationLabel, type_info: FieldTypeInfo) -> FieldMeta:
        return FieldMeta(
            field_type=FieldType.CLASSIFICATION_LABEL,
            config=ClassificationLabelConfig(
                allowed_labels=desc.allowed_labels
            ),
            type_info=type_info,
            converter=desc.converter
        )

    @staticmethod
    def _to_ocr_regex_meta(desc: OcrRegex, type_info: FieldTypeInfo) -> FieldMeta:
        return FieldMeta(
            field_type=FieldType.OCR_REGEX,
            config=OcrRegexConfig(
            ),
            type_info=type_info,
            converter=desc.converter
        )
        
    _handlers: Dict[Type[FieldBase], Callable[[Union[ClassificationLabel | OcrRegex | EntityRef], FieldTypeInfo], FieldMeta]] = {
        EntityRef: _create_entity_ref_meta,
        ClassificationLabel: _create_classification_label_meta,
        OcrRegex: _to_ocr_regex_meta
    }
    
    
class EntityMetaFactory:
    @staticmethod
    def create(
        cls: Type[EntityBase],
        detection_class: Optional[str], 
        fixed_point: Optional[Tuple[float, float]],
        parent_selector: Union[str | ParentSelectorType | None]
    ) -> EntityMeta:
        coordinates_provider = None
        entity_type = EntityType.PHYSICAL

        if fixed_point and detection_class:
            raise ValueError("An entity cannot have both a fixed_point and a detection_class.")
        elif fixed_point:
            coordinates_provider = StaticCoordinatesProvider(fixed_point)
        elif detection_class:
            coordinates_provider = DetectionCoordinatesProvider()
        else:
            entity_type = EntityType.LOGICAL

        source = EntitySource(entity_type, coordinates_provider)
        
        if isinstance(parent_selector, str):
            parent_selector = ParentSelectorType.from_value(parent_selector)
        parent_selector_strategy = ParentSelectorFactory.create(parent_selector)
        
        fields_meta = {}
        hints = get_type_hints(cls, globalns=sys.modules[cls.__module__].__dict__)
        
        for attr_name, type_hint in hints.items():
            attr = getattr(cls, attr_name, None)
            type_info = FieldTypeInfo.from_type_hint(type_hint)
            
            if attr is None and issubclass(type_info.base_type, EntityBase):
                attr = EntityRef(type_info.base_type)
                attr.__set_name__(cls, attr_name)
                setattr(cls, attr_name, attr)
                
            elif isinstance(attr, FieldBase):
                if isinstance(attr, EntityRef) and attr.entity_cls is None:
                    attr.entity_cls = type_info.base_type
                
            if isinstance(attr, FieldBase):
                fields_meta[attr_name] = FieldMetaFactory.create(attr, type_info)
            
        return EntityMeta(
            source=source,
            parent_selector=parent_selector_strategy,
            fields=fields_meta
        )