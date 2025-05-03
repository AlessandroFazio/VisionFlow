from typing import Type, get_type_hints

from anthropic import BaseModel
from pydantic import create_model

from visionflow.core.entity.reflection.meta import EntityMeta


class EntityBase:
    __vf_pydantic_model__: Type[BaseModel] = None
    __vf_meta__: EntityMeta = EntityMeta()

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
    
    def __init__(self, **kwargs) -> None:
        field_names = set(self.__vf_meta__.fields.keys())
        for name, obj in kwargs.items():
            if name not in field_names:
                raise ValueError("")
            setattr(self, name, obj)

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