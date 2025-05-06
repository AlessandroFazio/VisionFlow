from typing import Callable, Optional, Tuple, Type, Union
from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.parent_selector.config import ParentSelectorType
from visionflow.core.entity.reflection.meta.factory import EntityMetaFactory
from visionflow.typing.utils import ClassFactory


def entity(
    detection_class: Optional[str]=None, 
    fixed_point: Optional[Tuple[float, float]]=None,
    parent_selector: Union[str | ParentSelectorType]="nearest",
) -> Callable[[Type[EntityBase]], Type[EntityBase]]:
    def decorator(cls: type) -> Type[EntityBase]:
        if not issubclass(cls, EntityBase):
            cls = ClassFactory.ensure_base(cls, EntityBase)
        cls.__vf_meta__ = EntityMetaFactory.create(cls, detection_class, fixed_point, parent_selector)
        return cls
    return decorator