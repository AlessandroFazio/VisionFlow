import sys
from typing import Callable, Optional, Tuple, Type, Union, get_type_hints
from visionflow.core.entity.reflection.descriptors import EntityRef, FieldBase
from visionflow.core.pipeline.utils.providers import DetectionCoordinatesProvider, StaticCoordinatesProvider
from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.parent_selector.config import ParentSelectorType
from visionflow.core.entity.parent_selector.factory import ParentSelectorFactory
from visionflow.core.entity.reflection.meta import EntityMeta, EntitySource, EntityType, FieldTypeInfo


def entity(
    detection_class: Optional[str]=None, 
    fixed_point: Optional[Tuple[float, float]]=None,
    parent_selector: Union[str | ParentSelectorType]="nearest",
) -> Callable[[Type[EntityBase]], Type[EntityBase]]:
    def decorator(cls: type) -> Type[EntityBase]:
        nonlocal detection_class, fixed_point, parent_selector
        
        if not issubclass(cls, EntityBase):
            old = cls
            cls = type(cls.__name__, (cls, EntityBase), dict(cls.__dict__))
            cls.__module__ = old.__module__
            cls.__qualname__ = old.__qualname__

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
            meta = None
            
            if isinstance(attr, FieldBase):
                if isinstance(attr, EntityRef) and attr.entity_cls is None:
                    attr.entity_cls = type_info.base_type
                meta = attr._to_field_meta(type_info)
            
            elif attr is None and issubclass(type_info.base_type, EntityBase):
                attr = EntityRef(type_info.base_type)
                attr.__set_name__(cls, attr_name)
                setattr(cls, attr_name, attr)
                meta = attr._to_field_meta(type_info)
                
            if meta:
                fields_meta[attr_name] = meta
            
        meta = EntityMeta(
            source=source,
            parent_selector=parent_selector_strategy,
            fields=fields_meta
        )
        cls.__vf_meta__ = meta

        return cls
    return decorator
    