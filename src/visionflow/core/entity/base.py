from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints

from anthropic import BaseModel
from pydantic import create_model

from visionflow.core.pipeline.base import Exchange


@dataclass
class OcrRegexConfig:
    pattern: str

@dataclass
class ClassificationLabelConfig:
    single: bool
    allowed_labels: Optional[List[str]] = None

class FieldType(Enum):
    OCR_REGEX = 0
    CLASSIFICATION_LABEL = 1
    ENTITY_REF = 2

@dataclass
class FieldMeta:
    field_type: FieldType
    type_hint: Any
    config: Union[ClassificationLabelConfig | OcrRegexConfig | None] = None
    priority: int=0
    converter: Optional[Callable[[Any], Any]]=None

@dataclass
class EntityMeta:
    branch_binding: Optional[str] = None
    fields: Dict[str, FieldMeta] = field(default_factory=dict)


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


class FieldBase(ABC):
    def __init__(
        self, 
        converter: Optional[Callable[[Any], Any]]=None, 
        priority: int=0
    ) -> None:
        self.converter = converter
        self.priority = priority
        self.name = None

    @abstractmethod
    def __to_field_meta(self, type_hint: Any) -> FieldMeta:
        pass
    
    def __field_type_hint(self, owner: Type[EntityBase], name: str) -> Any:
        raw = owner.__annotations__.get(name)
        hints = get_type_hints(owner)
        return hints.get(name, raw)

    def __register_field_meta(self, owner: Type[EntityBase], name: str) -> None:
        type_hint = self.__field_type_hint(owner, name)
        owner.__vf_meta__.fields[name] = self.__to_field_meta(type_hint)

    def __set_name__(self, owner: Type[EntityBase], name: str) -> None:
        self.name = name
        self.__register_field_meta(owner, name)

    def __set__(self, instance: EntityBase, value: Any) -> None:
        if instance is None:
            return        
        instance.__dict__[self.name] = value

    def __get__(self, instance: EntityBase, owner: Type[EntityBase]) -> Any:
        return instance.__dict__.get(self.name) if instance else self

class ClassificationLabelField(FieldBase):
    def __init__(
        self, 
        single: bool=True, 
        allowed_labels: List[str]=None, 
        converter: Optional[Callable[[Any], Any]]=None, 
        priority: int=0
    ) -> None:
        self.single = single
        self.allowed_labels = allowed_labels or []
        super().__init__(converter=converter, priority=priority)
    
    def __to_field_meta(self, type_hint: Any) -> FieldMeta:
        return FieldMeta(
            field_type=FieldType.CLASSIFICATION_LABEL,
            config=ClassificationLabelConfig(
                single=self.single, 
                allowed_labels=self.allowed_labels
            ),
            type_hint=type_hint,
            converter=self.converter,
            priority=self.priority
        )

class OcrRegexField(FieldBase):
    def __init__(
        self, 
        pattern: str,
        converter: Optional[Callable[[Any], Any]]=None, 
        priority: int=0
    ) -> None:
        self.pattern = pattern
        super().__init__(converter=converter, priority=priority)

    def __to_field_meta(self, type_hint: Any) -> FieldMeta:
        return FieldMeta(
            field_type=FieldType.OCR_REGEX,
            config=OcrRegexConfig(
                pattern=self.pattern
            ),
            type_hint=type_hint,
            converter=self.converter,
            priority=self.priority
        )

class EntityRefField(FieldBase):
    def __init__(
        self, 
        converter: Optional[Callable[[Any], Any]]=None, 
        priority: int=0
    ) -> None:
        super().__init__(converter=converter, priority=priority)

    def __to_field_meta(self, type_hint: Any) -> FieldMeta:
        return FieldMeta(
            field_type=FieldType.ENTITY_REF,
            type_hint=type_hint,
            converter=self.converter,
            priority=self.priority
        )


def branch_binding(name: str) -> Callable[[Type[EntityBase]], Type[EntityBase]]:
    def decorator(cls: Type[EntityBase]) -> Type[EntityBase]:
        cls.__vf_meta__.branch_binding = name
        return cls
    return decorator


class Card(EntityBase):
    value: str = ClassificationLabelField(allowed_labels=[])


@branch_binding(name="player_info")
class SeatState(EntityBase):
    empty: bool = OcrRegexField(pattern="", converter=...)

@branch_binding(branch="player_info")
class PokerPlayer(EntityBase):
    username: str = OcrRegexField(pattern)

class TableSeat(EntityBase):
    state: SeatState = EntityRefField
    player: PokerPlayer = EntityRefField()
    is_dealer: bool = EntityRefField()


class EntityFactory:
    @staticmethod
    def from_exchange(cls: Type[EntityBase], exchange: Exchange) -> EntityBase:
        kwargs = {}
        fields_meta = cls.__vf_meta__.fields
        
        for field_name, meta in fields_meta.items():
            field_type = meta.field_type
            if field_type == FieldType.OCR_REGEX:
                ...
            elif field_type == FieldType.CLASSIFICATION_LABEL:
                ...
            elif field_type == FieldType.ENTITY_REF:
                ...

        return cls(**kwargs)