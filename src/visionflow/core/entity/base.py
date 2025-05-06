from typing import TYPE_CHECKING, Optional, Tuple, Type, Union, get_type_hints

from anthropic import BaseModel
from pydantic import create_model

from visionflow.core.types import XyXyType

if TYPE_CHECKING:
    from visionflow.core.entity.reflection.meta import EntityMeta


class EntityBase:
    __vf_pydantic_model__: Type[BaseModel] = None
    __vf_meta__: "EntityMeta" = None

    @classmethod
    def __pydantic_model(cls) -> Type[BaseModel]:
        if cls.__vf_pydantic_model__ is not None:
            return cls.__vf_pydantic_model__

        hints = get_type_hints(cls)
        fields = {
            name: (hint, getattr(cls, name, ...))
            for name, hint in hints.items()
            if not name.startswith("_")
        }

        pyd_cls = create_model(
            cls.__name__ + "Schema",
            __config__ = cls.get_pydantic_config() if hasattr(cls, "get_pydantic_config") else None,
            **fields
        )

        cls.__vf_pydantic_model__ = pyd_cls
        return pyd_cls
    
    def __init__(
        self, 
        __vf_xy__: Optional[XyXyType]=None,
        __vf_img_shape__: Optional[Tuple[int, int]]=None,
        **kwargs
    ) -> None:
        field_names = set(self.__vf_meta__.fields.keys())
        for name, obj in kwargs.items():
            if name not in field_names:
                raise ValueError("")
            setattr(self, name, obj)
        self.__vf_xy__ = __vf_xy__
        self.__vf_img_shape__ = __vf_img_shape__
        self.__vf_xy_normed__ = (
            (__vf_xy__[0] / __vf_img_shape__[0], __vf_xy__[1] / __vf_img_shape__[1]) 
            if __vf_xy__ 
            else None
        )

    def to_pydantic(self) -> BaseModel:
        P = self.__class__.__pydantic_model()
        data = {}
        for k,v in vars(self).items():
            if k.startswith("_"):
                continue
            if isinstance(v, EntityBase):
                v = v.to_pydantic()
            data[k] = v
        return P(**data)

    def coordinates(self, normalized: bool=False) -> Union[Tuple[float, float] | Tuple[int, int] | None]:
        return self.__vf_xy_normed__ if normalized else self.__vf_xy__