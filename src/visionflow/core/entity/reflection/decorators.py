from typing import Callable, Optional, Tuple, Type, Union
from visionflow.core.common.coordinates.providers import DetectionCoordinatesProvider, StaticCoordinatesProvider
from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.parent_selector.config import ParentSelectorType
from visionflow.core.entity.parent_selector.factory import ParentSelectorFactory
from visionflow.core.entity.reflection.meta import EntityMeta, EntitySource, EntityType


def entity(
    type: Union[str | EntityType], 
    detection_class: Optional[str]=None, 
    fixed_point: Optional[Tuple[float, float]]=None,
    parent_selector: Union[str | ParentSelectorType | None]=None
) -> Callable[[Type[EntityBase]], Type[EntityBase]]:
    def decorator(entity_cls: Type[EntityBase]) -> Type[EntityBase]:
        nonlocal type, parent_selector
        if isinstance(type, str):
            type = EntityType[type]

        coordinates_provider = None

        if type == EntityType.PHYSICAL:
            if fixed_point and detection_class:
                raise ValueError("")
            elif fixed_point:
                coordinates_provider = StaticCoordinatesProvider(fixed_point)
            elif detection_class:
                coordinates_provider = DetectionCoordinatesProvider()
                branch_selector = branch_selector or detection_class

        source = EntitySource(type, coordinates_provider)

        parent_selector_strategy = None
        
        if isinstance(parent_selector, str):
            parent_selector = ParentSelectorType.from_value(parent_selector)
        
        if parent_selector:
            parent_selector_strategy = ParentSelectorFactory.create(parent_selector)

        meta = EntityMeta(
            source=source,
            parent_selector=parent_selector_strategy
        )
        entity_cls.__vf_meta__ = meta

        return entity_cls
    return decorator
    